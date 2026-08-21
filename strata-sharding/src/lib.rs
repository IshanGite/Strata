use parking_lot::Mutex;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use strata_consensus::{
    AppendEntriesReq, AppendEntriesResp, InstallSnapshotReq, InstallSnapshotResp, NodeId, RaftNode,
    RaftTransport, RequestVoteReq, RequestVoteResp, StateMachine, StateMachineError,
    TransportError,
};
use strata_storage::Storage;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct ShardId(pub u32);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoadReport {
    pub shard_loads: Vec<(ShardId, u64)>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum RebalanceOp {
    Move {
        shard: ShardId,
        from: NodeId,
        to: NodeId,
    },
    Split {
        shard: ShardId,
        new_shard: ShardId,
        split_key: Vec<u8>,
    },
    Merge {
        shard_a: ShardId,
        shard_b: ShardId,
        target: ShardId,
    },
}

pub trait ShardRouter: Send + Sync {
    fn shard_for_key(&self, key: &[u8]) -> ShardId;
    fn raft_group_for_shard(&self, shard: ShardId) -> Vec<NodeId>;
    fn rebalance_plan(&self, current_load: &LoadReport) -> Vec<RebalanceOp>;
}

// ----------------------------------------------------------------------
// Range Routing Table
// ----------------------------------------------------------------------
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RangeRoute {
    pub start_key: Vec<u8>,
    pub end_key: Vec<u8>, // Empty represents positive infinity
    pub shard_id: ShardId,
    pub raft_group: Vec<NodeId>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RoutingTable {
    pub routes: Vec<RangeRoute>,
}

impl RoutingTable {
    pub fn new() -> Self {
        Self { routes: Vec::new() }
    }

    pub fn propose_rebalance(
        &self,
        current_load: &LoadReport,
        all_nodes: &[NodeId],
        threshold: u64,
    ) -> Vec<RebalanceOp> {
        let mut node_shards: HashMap<NodeId, Vec<ShardId>> = HashMap::new();
        for &node in all_nodes {
            node_shards.insert(node, Vec::new());
        }
        for r in &self.routes {
            for &node in &r.raft_group {
                if let Some(shards) = node_shards.get_mut(&node) {
                    shards.push(r.shard_id);
                }
            }
        }

        let mut shard_load_map = HashMap::new();
        for &(shard, load) in &current_load.shard_loads {
            shard_load_map.insert(shard, load);
        }

        let mut node_loads: HashMap<NodeId, u64> = HashMap::new();
        for &node in all_nodes {
            let load = node_shards
                .get(&node)
                .cloned()
                .unwrap_or_default()
                .iter()
                .map(|s| shard_load_map.get(s).cloned().unwrap_or(1))
                .sum();
            node_loads.insert(node, load);
        }

        let mut ops = Vec::new();
        let mut temp_loads = node_loads.clone();
        let mut temp_shards = node_shards.clone();

        loop {
            let mut min_node = None;
            let mut max_node = None;
            let mut min_val = u64::MAX;
            let mut max_val = 0;

            for &node in all_nodes {
                let val = temp_loads.get(&node).cloned().unwrap_or(0);
                if val < min_val {
                    min_val = val;
                    min_node = Some(node);
                }
                if val > max_val {
                    max_val = val;
                    max_node = Some(node);
                }
            }

            let min_n = match min_node {
                Some(n) => n,
                None => break,
            };
            let max_n = match max_node {
                Some(n) => n,
                None => break,
            };

            if max_val - min_val <= threshold {
                break;
            }

            let candidate_shards = temp_shards.get(&max_n).cloned().unwrap_or_default();
            let mut moved = false;
            for s in candidate_shards {
                let target_has_shard = temp_shards
                    .get(&min_n)
                    .cloned()
                    .unwrap_or_default()
                    .contains(&s);
                if !target_has_shard {
                    let s_load = shard_load_map.get(&s).cloned().unwrap_or(1);
                    let diff_before = max_val - min_val;
                    let diff_after = if max_val >= min_val + 2 * s_load {
                        max_val - min_val - 2 * s_load
                    } else {
                        min_val + 2 * s_load - max_val
                    };

                    if diff_after < diff_before {
                        ops.push(RebalanceOp::Move {
                            shard: s,
                            from: max_n,
                            to: min_n,
                        });

                        *temp_loads.get_mut(&max_n).unwrap() -= s_load;
                        *temp_loads.get_mut(&min_n).unwrap() += s_load;

                        temp_shards.get_mut(&max_n).unwrap().retain(|&x| x != s);
                        temp_shards.get_mut(&min_n).unwrap().push(s);
                        moved = true;
                        break;
                    }
                }
            }

            if !moved {
                break;
            }
        }
        ops
    }
}

impl ShardRouter for RoutingTable {
    fn shard_for_key(&self, key: &[u8]) -> ShardId {
        for r in &self.routes {
            let matches_start = key >= r.start_key.as_slice();
            let matches_end = r.end_key.is_empty() || key < r.end_key.as_slice();
            if matches_start && matches_end {
                return r.shard_id;
            }
        }
        self.routes
            .first()
            .map(|r| r.shard_id)
            .unwrap_or(ShardId(0))
    }

    fn raft_group_for_shard(&self, shard: ShardId) -> Vec<NodeId> {
        for r in &self.routes {
            if r.shard_id == shard {
                return r.raft_group.clone();
            }
        }
        Vec::new()
    }

    fn rebalance_plan(&self, current_load: &LoadReport) -> Vec<RebalanceOp> {
        let mut nodes = HashSet::new();
        for r in &self.routes {
            for &n in &r.raft_group {
                nodes.insert(n);
            }
        }
        let all_nodes: Vec<NodeId> = nodes.into_iter().collect();
        self.propose_rebalance(current_load, &all_nodes, 1)
    }
}

// ----------------------------------------------------------------------
// Meta Command (Shard 0 Commands)
// ----------------------------------------------------------------------
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum MetaCommand {
    UpdateRoute(RangeRoute),
    RemoveRoute(ShardId),
}

// ----------------------------------------------------------------------
// Unified Shard State Machine
// ----------------------------------------------------------------------
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct LockInfo {
    pub primary: Vec<u8>,
    pub ts: strata_storage::HlcTimestamp,
    pub ttl: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum PrewriteError {
    WriteConflict(strata_storage::HlcTimestamp),
    LockConflict {
        primary: Vec<u8>,
        ts: strata_storage::HlcTimestamp,
    },
}

pub fn lock_key(key: &[u8]) -> Vec<u8> {
    let mut k = Vec::with_capacity(1 + key.len());
    k.push(b'l');
    k.extend_from_slice(key);
    k
}

pub fn data_key(key: &[u8]) -> Vec<u8> {
    let mut k = Vec::with_capacity(1 + key.len());
    k.push(b'd');
    k.extend_from_slice(key);
    k
}

pub fn write_key(key: &[u8]) -> Vec<u8> {
    let mut k = Vec::with_capacity(1 + key.len());
    k.push(b'w');
    k.extend_from_slice(key);
    k
}

pub fn error_key(key: &[u8], start_ts: strata_storage::HlcTimestamp) -> Vec<u8> {
    let mut k = Vec::with_capacity(1 + key.len() + 12);
    k.push(b'e');
    k.extend_from_slice(key);
    k.extend_from_slice(&start_ts.physical.to_be_bytes());
    k.extend_from_slice(&start_ts.logical.to_be_bytes());
    k
}

pub fn commit_key(key: &[u8], start_ts: strata_storage::HlcTimestamp) -> Vec<u8> {
    let mut k = Vec::with_capacity(1 + key.len() + 12);
    k.push(b'c');
    k.extend_from_slice(key);
    k.extend_from_slice(&start_ts.physical.to_be_bytes());
    k.extend_from_slice(&start_ts.logical.to_be_bytes());
    k
}

pub fn rollback_key(key: &[u8], start_ts: strata_storage::HlcTimestamp) -> Vec<u8> {
    let mut k = Vec::with_capacity(1 + key.len() + 12);
    k.push(b'r');
    k.extend_from_slice(key);
    k.extend_from_slice(&start_ts.physical.to_be_bytes());
    k.extend_from_slice(&start_ts.logical.to_be_bytes());
    k
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ShardCommand {
    Put {
        key: Vec<u8>,
        value: Vec<u8>,
        ts: strata_storage::HlcTimestamp,
    },
    Delete {
        key: Vec<u8>,
        ts: strata_storage::HlcTimestamp,
    },
    Split {
        new_shard_id: ShardId,
        split_key: Vec<u8>,
    },
    Merge {
        target_shard_id: ShardId,
    },
    TxnPrewrite {
        key: Vec<u8>,
        value: Option<Vec<u8>>,
        primary: Vec<u8>,
        start_ts: strata_storage::HlcTimestamp,
        ttl: u64,
    },
    TxnCommit {
        key: Vec<u8>,
        start_ts: strata_storage::HlcTimestamp,
        commit_ts: strata_storage::HlcTimestamp,
        is_primary: bool,
    },
    TxnRollback {
        key: Vec<u8>,
        start_ts: strata_storage::HlcTimestamp,
    },
}

pub struct ShardStateMachine {
    pub shard_id: ShardId,
    pub storage: Arc<Mutex<Option<strata_storage::LsmStorage>>>,
    pub db_dir: PathBuf,
    pub table: Arc<Mutex<RoutingTable>>,
}

impl ShardStateMachine {
    pub fn open(shard_id: ShardId, db_dir: PathBuf, table: Arc<Mutex<RoutingTable>>) -> Self {
        let storage = if shard_id.0 == 0 {
            None
        } else {
            let lsm = strata_storage::LsmStorage::open(&db_dir, 1024 * 1024, 0.01).unwrap();
            let split_file = db_dir.with_file_name(format!("split_shard_{}.bin", shard_id.0));
            let merge_file = db_dir.with_file_name(format!("merge_shard_{}.bin", shard_id.0));

            let mut init_data = HashMap::new();
            if split_file.exists() {
                if let Ok(bytes) = std::fs::read(&split_file) {
                    if let Ok(data) = bincode::deserialize::<HashMap<Vec<u8>, Vec<u8>>>(&bytes) {
                        init_data = data;
                    }
                }
                let _ = std::fs::remove_file(&split_file);
            }
            if merge_file.exists() {
                if let Ok(bytes) = std::fs::read(&merge_file) {
                    if let Ok(data) = bincode::deserialize::<HashMap<Vec<u8>, Vec<u8>>>(&bytes) {
                        init_data = data;
                    }
                }
                let _ = std::fs::remove_file(&merge_file);
            }

            let ts = strata_storage::HlcTimestamp {
                physical: 0,
                logical: 0,
            };
            for (k, v) in init_data {
                lsm.put(&k, &v, ts).unwrap();
            }
            Some(lsm)
        };

        Self {
            shard_id,
            storage: Arc::new(Mutex::new(storage)),
            db_dir,
            table,
        }
    }
}

impl StateMachine for ShardStateMachine {
    fn apply(&self, command: &[u8]) -> Result<Vec<u8>, StateMachineError> {
        if self.shard_id.0 == 0 {
            let cmd: MetaCommand = bincode::deserialize(command)
                .map_err(|e| StateMachineError::ApplyFailed(e.to_string()))?;
            let mut table = self.table.lock();
            match cmd {
                MetaCommand::UpdateRoute(route) => {
                    table.routes.retain(|r| r.shard_id != route.shard_id);
                    table.routes.push(route);
                    table.routes.sort_by(|a, b| a.start_key.cmp(&b.start_key));
                }
                MetaCommand::RemoveRoute(shard_id) => {
                    table.routes.retain(|r| r.shard_id != shard_id);
                }
            }
            Ok(Vec::new())
        } else {
            let cmd: ShardCommand = bincode::deserialize(command)
                .map_err(|e| StateMachineError::ApplyFailed(e.to_string()))?;

            let mut opt = self.storage.lock();
            if opt.is_none() {
                return Err(StateMachineError::ApplyFailed(
                    "Storage not initialized".to_string(),
                ));
            }
            let storage = opt.as_mut().unwrap();

            match cmd {
                ShardCommand::Put { key, value, ts } => {
                    storage
                        .put(&key, &value, ts)
                        .map_err(|e| StateMachineError::ApplyFailed(e.to_string()))?;
                    Ok(value)
                }
                ShardCommand::Delete { key, ts } => {
                    storage
                        .delete(&key, ts)
                        .map_err(|e| StateMachineError::ApplyFailed(e.to_string()))?;
                    Ok(Vec::new())
                }
                ShardCommand::Split {
                    new_shard_id,
                    split_key,
                } => {
                    let max_ts = strata_storage::HlcTimestamp {
                        physical: u64::MAX,
                        logical: u32::MAX,
                    };
                    let iter = storage
                        .scan(&split_key, &[], max_ts)
                        .map_err(|e| StateMachineError::ApplyFailed(e.to_string()))?;

                    let mut split_data = HashMap::new();
                    for (k, v) in iter {
                        split_data.insert(k.clone(), v.clone());
                        let _ = storage.delete(&k, max_ts);
                    }

                    let split_file = self
                        .db_dir
                        .with_file_name(format!("split_shard_{}.bin", new_shard_id.0));
                    let bytes = bincode::serialize(&split_data).unwrap();
                    std::fs::write(split_file, bytes).unwrap();

                    Ok(Vec::new())
                }
                ShardCommand::Merge { target_shard_id } => {
                    let max_ts = strata_storage::HlcTimestamp {
                        physical: u64::MAX,
                        logical: u32::MAX,
                    };
                    let iter = storage
                        .scan(&[], &[], max_ts)
                        .map_err(|e| StateMachineError::ApplyFailed(e.to_string()))?;

                    let mut merge_data = HashMap::new();
                    for (k, v) in iter {
                        merge_data.insert(k, v);
                    }

                    let merge_file = self
                        .db_dir
                        .with_file_name(format!("merge_shard_{}.bin", target_shard_id.0));
                    let bytes = bincode::serialize(&merge_data).unwrap();
                    std::fs::write(merge_file, bytes).unwrap();

                    Ok(Vec::new())
                }
                ShardCommand::TxnPrewrite {
                    key,
                    value,
                    primary,
                    start_ts,
                    ttl,
                } => {
                    use strata_storage::Storage;

                    // 1. Write-Write conflict check
                    let wk = write_key(&key);
                    if let Ok(Some(write_bytes)) = storage.get(
                        &wk,
                        strata_storage::HlcTimestamp {
                            physical: u64::MAX,
                            logical: u32::MAX,
                        },
                    ) {
                        if let Ok((_w_start, w_commit)) = bincode::deserialize::<(
                            strata_storage::HlcTimestamp,
                            strata_storage::HlcTimestamp,
                        )>(&write_bytes)
                        {
                            if w_commit >= start_ts {
                                let ek = error_key(&key, start_ts);
                                let err_bytes =
                                    bincode::serialize(&PrewriteError::WriteConflict(w_commit))
                                        .unwrap();
                                let _ = storage.put(&ek, &err_bytes, start_ts);
                                return Ok(Vec::new());
                            }
                        }
                    }

                    // 2. Lock conflict check
                    let lk = lock_key(&key);
                    if let Ok(Some(lock_bytes)) = storage.get(
                        &lk,
                        strata_storage::HlcTimestamp {
                            physical: u64::MAX,
                            logical: u32::MAX,
                        },
                    ) {
                        if let Ok(lock_info) = bincode::deserialize::<LockInfo>(&lock_bytes) {
                            let ek = error_key(&key, start_ts);
                            let err_bytes = bincode::serialize(&PrewriteError::LockConflict {
                                primary: lock_info.primary.clone(),
                                ts: lock_info.ts,
                            })
                            .unwrap();
                            let _ = storage.put(&ek, &err_bytes, start_ts);
                            return Ok(Vec::new());
                        }
                    }

                    // 3. Write Lock
                    let lock_info = LockInfo {
                        primary,
                        ts: start_ts,
                        ttl,
                    };
                    let lock_bytes = bincode::serialize(&lock_info).unwrap();
                    storage
                        .put(&lk, &lock_bytes, start_ts)
                        .map_err(|e| StateMachineError::ApplyFailed(e.to_string()))?;

                    // 4. Write Data
                    let dk = data_key(&key);
                    let val_bytes = bincode::serialize(&value).unwrap();
                    storage
                        .put(&dk, &val_bytes, start_ts)
                        .map_err(|e| StateMachineError::ApplyFailed(e.to_string()))?;

                    Ok(Vec::new())
                }
                ShardCommand::TxnCommit {
                    key,
                    start_ts,
                    commit_ts,
                    is_primary,
                } => {
                    use strata_storage::Storage;
                    let lk = lock_key(&key);
                    if let Ok(Some(lock_bytes)) = storage.get(
                        &lk,
                        strata_storage::HlcTimestamp {
                            physical: u64::MAX,
                            logical: u32::MAX,
                        },
                    ) {
                        if let Ok(lock_info) = bincode::deserialize::<LockInfo>(&lock_bytes) {
                            if lock_info.ts == start_ts {
                                storage
                                    .delete(&lk, commit_ts)
                                    .map_err(|e| StateMachineError::ApplyFailed(e.to_string()))?;

                                let wk = write_key(&key);
                                let val = bincode::serialize(&(start_ts, commit_ts)).unwrap();
                                storage
                                    .put(&wk, &val, commit_ts)
                                    .map_err(|e| StateMachineError::ApplyFailed(e.to_string()))?;

                                if is_primary {
                                    let ck = commit_key(&key, start_ts);
                                    let ck_val = bincode::serialize(&commit_ts).unwrap();
                                    storage.put(&ck, &ck_val, commit_ts).map_err(|e| {
                                        StateMachineError::ApplyFailed(e.to_string())
                                    })?;
                                }
                            }
                        }
                    }
                    Ok(Vec::new())
                }
                ShardCommand::TxnRollback { key, start_ts } => {
                    use strata_storage::Storage;
                    let lk = lock_key(&key);
                    if let Ok(Some(lock_bytes)) = storage.get(
                        &lk,
                        strata_storage::HlcTimestamp {
                            physical: u64::MAX,
                            logical: u32::MAX,
                        },
                    ) {
                        if let Ok(lock_info) = bincode::deserialize::<LockInfo>(&lock_bytes) {
                            if lock_info.ts == start_ts {
                                storage
                                    .delete(&lk, start_ts)
                                    .map_err(|e| StateMachineError::ApplyFailed(e.to_string()))?;
                            }
                        }
                    }
                    let rk = rollback_key(&key, start_ts);
                    storage
                        .put(&rk, &[], start_ts)
                        .map_err(|e| StateMachineError::ApplyFailed(e.to_string()))?;
                    Ok(Vec::new())
                }
            }
        }
    }

    fn snapshot(&self) -> Result<Vec<u8>, StateMachineError> {
        if self.shard_id.0 == 0 {
            let table = self.table.lock();
            bincode::serialize(&*table)
                .map_err(|e| StateMachineError::SnapshotFailed(e.to_string()))
        } else {
            let opt = self.storage.lock();
            if opt.is_none() {
                return Ok(Vec::new());
            }
            let storage = opt.as_ref().unwrap();
            let max_ts = strata_storage::HlcTimestamp {
                physical: u64::MAX,
                logical: u32::MAX,
            };
            let iter = storage
                .scan(&[], &[], max_ts)
                .map_err(|e| StateMachineError::SnapshotFailed(e.to_string()))?;
            let mut data = HashMap::new();
            for (k, v) in iter {
                data.insert(k, v);
            }
            bincode::serialize(&data).map_err(|e| StateMachineError::SnapshotFailed(e.to_string()))
        }
    }

    fn restore(&self, snapshot: &[u8]) -> Result<(), StateMachineError> {
        if self.shard_id.0 == 0 {
            let table: RoutingTable = bincode::deserialize(snapshot)
                .map_err(|e| StateMachineError::RestoreFailed(e.to_string()))?;
            *self.table.lock() = table;
            Ok(())
        } else {
            let data: HashMap<Vec<u8>, Vec<u8>> = bincode::deserialize(snapshot)
                .map_err(|e| StateMachineError::RestoreFailed(e.to_string()))?;

            let mut opt = self.storage.lock();
            if opt.is_none() {
                let fresh = strata_storage::LsmStorage::open(&self.db_dir, 1024 * 1024, 0.01)
                    .map_err(|e| StateMachineError::RestoreFailed(e.to_string()))?;
                *opt = Some(fresh);
            }
            let storage = opt.as_mut().unwrap();

            let _ = std::fs::remove_dir_all(&self.db_dir);
            let fresh = strata_storage::LsmStorage::open(&self.db_dir, 1024 * 1024, 0.01)
                .map_err(|e| StateMachineError::RestoreFailed(e.to_string()))?;

            let max_ts = strata_storage::HlcTimestamp {
                physical: 0,
                logical: 0,
            };
            for (k, v) in data {
                fresh
                    .put(&k, &v, max_ts)
                    .map_err(|e| StateMachineError::RestoreFailed(e.to_string()))?;
            }
            *storage = fresh;
            Ok(())
        }
    }
}

pub type ShardNode<T = MultiRaftTransport> = RaftNode<ShardStateMachine, T>;
pub type ShardMap<T = MultiRaftTransport> = HashMap<ShardId, Arc<ShardNode<T>>>;

pub struct MultiRaftNode<T: RaftTransport + 'static = MultiRaftTransport> {
    pub node_id: NodeId,
    pub db_dir: PathBuf,
    pub shards: Arc<Mutex<ShardMap<T>>>,
    pub sms: Arc<Mutex<HashMap<ShardId, Arc<ShardStateMachine>>>>,
    pub transport: Arc<T>,
    pub table: Arc<Mutex<RoutingTable>>,
}

impl<T: RaftTransport + 'static> MultiRaftNode<T> {
    pub fn new(node_id: NodeId, db_dir: PathBuf, transport: Arc<T>) -> Self {
        Self {
            node_id,
            db_dir,
            shards: Arc::new(Mutex::new(HashMap::new())),
            sms: Arc::new(Mutex::new(HashMap::new())),
            transport,
            table: Arc::new(Mutex::new(RoutingTable::new())),
        }
    }

    pub fn start_shard(&self, shard_id: ShardId, peers: Vec<NodeId>) {
        let shard_dir = self.db_dir.join(format!("shard_{}", shard_id.0));
        let sm = Arc::new(ShardStateMachine::open(
            shard_id,
            shard_dir,
            self.table.clone(),
        ));
        let wal_path = self.db_dir.join(format!("wal_shard_{}.log", shard_id.0));

        let node = Arc::new(
            RaftNode::new(
                shard_id.0,
                self.node_id,
                peers,
                wal_path,
                sm.clone(),
                self.transport.clone(),
            )
            .unwrap(),
        );

        self.shards.lock().insert(shard_id, node);
        self.sms.lock().insert(shard_id, sm);
    }

    pub fn stop_shard(&self, shard_id: ShardId) {
        let node = self.shards.lock().remove(&shard_id);
        if let Some(n) = node {
            n.shutdown();
        }
        self.sms.lock().remove(&shard_id);
    }

    pub fn tick(&self) {
        let shards = self.shards.lock();
        for shard_node in shards.values() {
            shard_node.tick();
        }
    }
}

// ----------------------------------------------------------------------
// Multi-Raft Chaos Network Simulator
// ----------------------------------------------------------------------
pub enum NetworkPayload {
    RequestVote(RequestVoteReq),
    AppendEntries(AppendEntriesReq),
    InstallSnapshot(InstallSnapshotReq),
}

impl NetworkPayload {
    pub fn shard_id(&self) -> ShardId {
        match self {
            NetworkPayload::RequestVote(req) => ShardId(req.shard_id),
            NetworkPayload::AppendEntries(req) => ShardId(req.shard_id),
            NetworkPayload::InstallSnapshot(req) => ShardId(req.shard_id),
        }
    }
}

pub enum NetworkResponse {
    RequestVote(RequestVoteResp),
    AppendEntries(AppendEntriesResp),
    InstallSnapshot(InstallSnapshotResp),
}

pub struct PendingMsg {
    pub from: NodeId,
    pub to: NodeId,
    pub payload: NetworkPayload,
    pub reply_tx: std::sync::mpsc::Sender<Result<NetworkResponse, TransportError>>,
}

pub struct MultiRaftTransport {
    pub node_id: NodeId,
    pub network: Arc<Mutex<ChaosNetwork>>,
}

impl RaftTransport for MultiRaftTransport {
    fn send_request_vote(
        &self,
        to: NodeId,
        req: RequestVoteReq,
    ) -> Result<RequestVoteResp, TransportError> {
        let (tx, rx) = std::sync::mpsc::channel();
        let msg = PendingMsg {
            from: self.node_id,
            to,
            payload: NetworkPayload::RequestVote(req),
            reply_tx: tx,
        };
        self.network.lock().enqueue(msg);
        match rx.recv().map_err(|_| TransportError::Timeout)? {
            Ok(NetworkResponse::RequestVote(resp)) => Ok(resp),
            Ok(_) => Err(TransportError::Other("Invalid response type".to_string())),
            Err(e) => Err(e),
        }
    }

    fn send_append_entries(
        &self,
        to: NodeId,
        req: AppendEntriesReq,
    ) -> Result<AppendEntriesResp, TransportError> {
        let (tx, rx) = std::sync::mpsc::channel();
        let msg = PendingMsg {
            from: self.node_id,
            to,
            payload: NetworkPayload::AppendEntries(req),
            reply_tx: tx,
        };
        self.network.lock().enqueue(msg);
        match rx.recv().map_err(|_| TransportError::Timeout)? {
            Ok(NetworkResponse::AppendEntries(resp)) => Ok(resp),
            Ok(_) => Err(TransportError::Other("Invalid response type".to_string())),
            Err(e) => Err(e),
        }
    }

    fn send_install_snapshot(
        &self,
        to: NodeId,
        req: InstallSnapshotReq,
    ) -> Result<InstallSnapshotResp, TransportError> {
        let (tx, rx) = std::sync::mpsc::channel();
        let msg = PendingMsg {
            from: self.node_id,
            to,
            payload: NetworkPayload::InstallSnapshot(req),
            reply_tx: tx,
        };
        self.network.lock().enqueue(msg);
        match rx.recv().map_err(|_| TransportError::Timeout)? {
            Ok(NetworkResponse::InstallSnapshot(resp)) => Ok(resp),
            Ok(_) => Err(TransportError::Other("Invalid response type".to_string())),
            Err(e) => Err(e),
        }
    }
}

pub struct ChaosNetwork {
    pub node_servers: HashMap<NodeId, Arc<MultiRaftNode>>,
    pub partitions: HashSet<(NodeId, NodeId)>,
    pub pending_msgs: Mutex<Vec<PendingMsg>>,
    pub delayed_msgs: Vec<(u64, PendingMsg)>,
    pub loss_rate: f64,
    pub delay_range: std::ops::Range<u64>,
    pub virtual_time: u64,
    pub rng: rand::rngs::StdRng,
}

impl ChaosNetwork {
    pub fn new(seed: u64) -> Self {
        Self {
            node_servers: HashMap::new(),
            partitions: HashSet::new(),
            pending_msgs: Mutex::new(Vec::new()),
            delayed_msgs: Vec::new(),
            loss_rate: 0.0,
            delay_range: 0..1,
            virtual_time: 0,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    pub fn set_partition(&mut self, node1: NodeId, node2: NodeId) {
        self.partitions.insert((node1, node2));
        self.partitions.insert((node2, node1));
    }

    pub fn heal_partitions(&mut self) {
        self.partitions.clear();
    }

    pub fn enqueue(&self, msg: PendingMsg) {
        self.pending_msgs.lock().push(msg);
    }

    pub fn tick(&mut self) {
        let mut new_pending = {
            let mut guard = self.pending_msgs.lock();
            std::mem::take(&mut *guard)
        };

        for msg in new_pending.drain(..) {
            let is_partitioned = self.partitions.contains(&(msg.from, msg.to))
                || self.partitions.contains(&(msg.to, msg.from));
            if is_partitioned {
                let _ = msg.reply_tx.send(Err(TransportError::Timeout));
                continue;
            }

            if self.rng.gen_bool(self.loss_rate) {
                let _ = msg.reply_tx.send(Err(TransportError::Timeout));
                continue;
            }

            let delay = if self.delay_range.end > self.delay_range.start {
                self.rng.gen_range(self.delay_range.clone())
            } else {
                0
            };
            self.delayed_msgs.push((self.virtual_time + delay, msg));
        }

        self.virtual_time += 1;

        for server in self.node_servers.values() {
            let shards = server.shards.lock();
            for node in shards.values() {
                node.tick();
            }
        }

        let mut remaining = Vec::new();
        for (deliver_at, msg) in self.delayed_msgs.drain(..) {
            if deliver_at <= self.virtual_time {
                let is_partitioned = self.partitions.contains(&(msg.from, msg.to))
                    || self.partitions.contains(&(msg.to, msg.from));
                if is_partitioned {
                    let _ = msg.reply_tx.send(Err(TransportError::Timeout));
                    continue;
                }

                if let Some(target_server) = self.node_servers.get(&msg.to) {
                    let target_server_clone = target_server.clone();
                    let shard_id = msg.payload.shard_id();
                    let target_raft_node = {
                        let shards = target_server_clone.shards.lock();
                        shards.get(&shard_id).cloned()
                    };

                    if let Some(target_node) = target_raft_node {
                        tokio::task::block_in_place(move || {
                            futures::executor::block_on(async {
                                match msg.payload {
                                    NetworkPayload::RequestVote(req) => {
                                        let resp = target_node.handle_request_vote_rpc(req).await;
                                        let _ = msg
                                            .reply_tx
                                            .send(Ok(NetworkResponse::RequestVote(resp)));
                                    }
                                    NetworkPayload::AppendEntries(req) => {
                                        let resp = target_node.handle_append_entries_rpc(req).await;
                                        let _ = msg
                                            .reply_tx
                                            .send(Ok(NetworkResponse::AppendEntries(resp)));
                                    }
                                    NetworkPayload::InstallSnapshot(req) => {
                                        let resp =
                                            target_node.handle_install_snapshot_rpc(req).await;
                                        let _ = msg
                                            .reply_tx
                                            .send(Ok(NetworkResponse::InstallSnapshot(resp)));
                                    }
                                }
                            });
                        });
                    } else {
                        let _ = msg.reply_tx.send(Err(TransportError::ConnectionRefused));
                    }
                } else {
                    let _ = msg.reply_tx.send(Err(TransportError::ConnectionRefused));
                }
            } else {
                remaining.push((deliver_at, msg));
            }
        }
        self.delayed_msgs = remaining;
    }
}
