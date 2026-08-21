use std::fmt;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
pub use strata_storage::HlcTimestamp;

use strata_consensus::{RaftTransport, Role};
use strata_sharding::{
    commit_key, data_key, error_key, lock_key, write_key, LockInfo, MultiRaftNode, PrewriteError,
    RoutingTable, ShardCommand, ShardRouter,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Mutation {
    Put(Vec<u8>, Vec<u8>),
    Delete(Vec<u8>),
}

#[derive(Debug)]
pub enum TxnError {
    LockConflict {
        key: Vec<u8>,
        primary: Vec<u8>,
        lock_ts: HlcTimestamp,
    },
    WriteConflict {
        key: Vec<u8>,
        conflict_ts: HlcTimestamp,
    },
    Aborted,
    Storage(strata_storage::StorageError),
    Other(String),
}

impl fmt::Display for TxnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TxnError::LockConflict {
                key,
                primary,
                lock_ts,
            } => {
                write!(
                    f,
                    "Lock conflict on key {:?}, primary keys {:?}, lock_ts: {}",
                    key, primary, lock_ts
                )
            }
            TxnError::WriteConflict { key, conflict_ts } => {
                write!(
                    f,
                    "Write conflict on key {:?}, conflict_ts: {}",
                    key, conflict_ts
                )
            }
            TxnError::Aborted => write!(f, "Transaction aborted"),
            TxnError::Storage(e) => write!(f, "Storage error: {}", e),
            TxnError::Other(e) => write!(f, "Txn error: {}", e),
        }
    }
}

impl std::error::Error for TxnError {}

pub trait TransactionCoordinator: Send + Sync {
    fn begin(&self) -> HlcTimestamp;
    fn prewrite(&self, txn_ts: HlcTimestamp, mutations: &[Mutation]) -> Result<(), TxnError>; // phase 1 of 2PC
    fn commit(&self, txn_ts: HlcTimestamp, commit_ts: HlcTimestamp) -> Result<(), TxnError>; // phase 2 of 2PC
    fn abort(&self, txn_ts: HlcTimestamp) -> Result<(), TxnError>;
}

pub struct Hlc {
    state: Mutex<HlcTimestamp>,
    clock: Box<dyn Fn() -> u64 + Send + Sync>,
}

impl Hlc {
    pub fn new(physical: u64, logical: u32) -> Self {
        Self {
            state: Mutex::new(HlcTimestamp { physical, logical }),
            clock: Box::new(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64
            }),
        }
    }

    pub fn new_with_clock(
        physical: u64,
        logical: u32,
        clock: Box<dyn Fn() -> u64 + Send + Sync>,
    ) -> Self {
        Self {
            state: Mutex::new(HlcTimestamp { physical, logical }),
            clock,
        }
    }

    pub fn local_event(&self) -> HlcTimestamp {
        let mut state = self.state.lock().unwrap();
        let pt = (self.clock)();
        if pt > state.physical {
            state.physical = pt;
            state.logical = 0;
        } else {
            state.logical += 1;
        }
        *state
    }

    pub fn receive(&self, msg_ts: HlcTimestamp) -> HlcTimestamp {
        let mut state = self.state.lock().unwrap();
        let pt = (self.clock)();
        let l_old = state.physical;
        let l_new = l_old.max(msg_ts.physical).max(pt);
        let c_new = if l_new == l_old && l_new == msg_ts.physical {
            state.logical.max(msg_ts.logical) + 1
        } else if l_new == l_old {
            state.logical + 1
        } else if l_new == msg_ts.physical {
            msg_ts.logical + 1
        } else {
            0
        };
        state.physical = l_new;
        state.logical = c_new;
        *state
    }

    pub fn read(&self) -> HlcTimestamp {
        *self.state.lock().unwrap()
    }
}

pub struct DistributedTxnCoordinator<
    T: RaftTransport + 'static = strata_sharding::MultiRaftTransport,
> {
    pub hlc: std::sync::Arc<Hlc>,
    pub node_servers: std::sync::Arc<
        parking_lot::Mutex<
            std::collections::HashMap<strata_consensus::NodeId, std::sync::Arc<MultiRaftNode<T>>>,
        >,
    >,
    pub table: std::sync::Arc<parking_lot::Mutex<RoutingTable>>,
    pub active_txns: parking_lot::Mutex<std::collections::HashMap<HlcTimestamp, Vec<Vec<u8>>>>,
}

impl<T: RaftTransport + 'static> DistributedTxnCoordinator<T> {
    pub fn new(
        hlc: std::sync::Arc<Hlc>,
        node_servers: std::sync::Arc<
            parking_lot::Mutex<
                std::collections::HashMap<
                    strata_consensus::NodeId,
                    std::sync::Arc<MultiRaftNode<T>>,
                >,
            >,
        >,
        table: std::sync::Arc<parking_lot::Mutex<RoutingTable>>,
    ) -> Self {
        Self {
            hlc,
            node_servers,
            table,
            active_txns: parking_lot::Mutex::new(std::collections::HashMap::new()),
        }
    }

    fn get_leader_node(
        &self,
        shard_id: strata_sharding::ShardId,
    ) -> Result<std::sync::Arc<strata_sharding::ShardNode<T>>, TxnError> {
        let raft_group = self.table.lock().raft_group_for_shard(shard_id);
        if raft_group.is_empty() {
            return Err(TxnError::Other(format!(
                "No raft group for shard {:?}",
                shard_id
            )));
        }
        let servers = self.node_servers.lock();
        for &node_id in &raft_group {
            if let Some(server) = servers.get(&node_id) {
                let shards = server.shards.lock();
                if let Some(shard_node) = shards.get(&shard_id) {
                    if shard_node.state.lock().role == Role::Leader {
                        return Ok(shard_node.clone());
                    }
                }
            }
        }
        Err(TxnError::Other(format!(
            "No leader found for shard {:?}",
            shard_id
        )))
    }

    pub async fn propose_command(&self, key: &[u8], cmd: ShardCommand) -> Result<(), TxnError> {
        let shard_id = self.table.lock().shard_for_key(key);

        let mut retries = 0;
        let shard_node = loop {
            match self.get_leader_node(shard_id) {
                Ok(node) => break node,
                Err(e) => {
                    if retries >= 100 {
                        return Err(e);
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    retries += 1;
                }
            }
        };

        let data = bincode::serialize(&cmd).map_err(|e| TxnError::Other(e.to_string()))?;
        let rx = shard_node.propose(data);
        let idx = match rx.await {
            Ok(Ok(idx)) => idx,
            Ok(Err(e)) => return Err(TxnError::Other(e)),
            Err(e) => return Err(TxnError::Other(e.to_string())),
        };

        let mut apply_retries = 0;
        loop {
            let last_applied = shard_node.state.lock().last_applied;
            if last_applied >= idx {
                break;
            }
            if apply_retries >= 200 {
                return Err(TxnError::Other(
                    "Timeout waiting for proposal to be applied".to_string(),
                ));
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            apply_retries += 1;
        }
        Ok(())
    }

    pub fn get_storage(
        &self,
        key: &[u8],
    ) -> Result<std::sync::Arc<parking_lot::Mutex<Option<strata_storage::LsmStorage>>>, TxnError>
    {
        let shard_id = self.table.lock().shard_for_key(key);
        let raft_group = self.table.lock().raft_group_for_shard(shard_id);
        if raft_group.is_empty() {
            return Err(TxnError::Other(format!(
                "No raft group for shard {:?}",
                shard_id
            )));
        }
        let servers = self.node_servers.lock();

        // Try leader first
        for &node_id in &raft_group {
            if let Some(server) = servers.get(&node_id) {
                let shards = server.shards.lock();
                if let Some(shard_node) = shards.get(&shard_id) {
                    if shard_node.state.lock().role == Role::Leader {
                        let sms = server.sms.lock();
                        if let Some(sm) = sms.get(&shard_id) {
                            return Ok(sm.storage.clone());
                        }
                    }
                }
            }
        }

        // Fallback to any node with initialized storage
        for &node_id in &raft_group {
            if let Some(server) = servers.get(&node_id) {
                let sms = server.sms.lock();
                if let Some(sm) = sms.get(&shard_id) {
                    let storage_opt = sm.storage.lock();
                    if storage_opt.is_some() {
                        drop(storage_opt);
                        return Ok(sm.storage.clone());
                    }
                }
            }
        }
        Err(TxnError::Other(format!(
            "No storage found for shard {:?}",
            shard_id
        )))
    }

    fn storage_get(
        &self,
        key: &[u8],
        query_key: &[u8],
        as_of: HlcTimestamp,
    ) -> Result<Option<Vec<u8>>, TxnError> {
        let storage_arc = self.get_storage(key)?;
        let guard = storage_arc.lock();
        if let Some(ref storage) = *guard {
            use strata_storage::Storage;
            storage.get(query_key, as_of).map_err(TxnError::Storage)
        } else {
            Err(TxnError::Other("Storage not initialized".to_string()))
        }
    }

    pub async fn get(&self, key: &[u8], as_of: HlcTimestamp) -> Result<Option<Vec<u8>>, TxnError> {
        let max_ts = HlcTimestamp {
            physical: u64::MAX,
            logical: u32::MAX,
        };
        let lk = lock_key(key);

        loop {
            let lock_opt = self.storage_get(key, &lk, max_ts)?;
            if let Some(lock_bytes) = lock_opt {
                if let Ok(lock_info) = bincode::deserialize::<LockInfo>(&lock_bytes) {
                    if lock_info.ts <= as_of {
                        let now_pt = self.hlc.read().physical;
                        if now_pt > lock_info.ts.physical + lock_info.ttl {
                            // Lock is stale! Resolve it.
                            let ck = commit_key(&lock_info.primary, lock_info.ts);
                            let commit_opt = self.storage_get(&lock_info.primary, &ck, max_ts)?;
                            if let Some(commit_bytes) = commit_opt {
                                if let Ok(commit_ts) =
                                    bincode::deserialize::<HlcTimestamp>(&commit_bytes)
                                {
                                    // Primary committed! Roll secondary forward.
                                    let commit_cmd = ShardCommand::TxnCommit {
                                        key: key.to_vec(),
                                        start_ts: lock_info.ts,
                                        commit_ts,
                                        is_primary: false,
                                    };
                                    self.propose_command(key, commit_cmd).await?;
                                    continue;
                                }
                            }
                            // Primary not committed. Roll back both primary and secondary.
                            let rollback_primary = ShardCommand::TxnRollback {
                                key: lock_info.primary.clone(),
                                start_ts: lock_info.ts,
                            };
                            let rollback_secondary = ShardCommand::TxnRollback {
                                key: key.to_vec(),
                                start_ts: lock_info.ts,
                            };
                            let _ = self
                                .propose_command(&lock_info.primary, rollback_primary)
                                .await;
                            let _ = self.propose_command(key, rollback_secondary).await;
                            continue;
                        } else {
                            // Wait for lock to resolve or timeout
                            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                            continue;
                        }
                    }
                }
            }
            break;
        }

        let wk = write_key(key);
        let write_opt = self.storage_get(key, &wk, as_of)?;
        if let Some(write_bytes) = write_opt {
            if let Ok((start_ts, _commit_ts)) =
                bincode::deserialize::<(HlcTimestamp, HlcTimestamp)>(&write_bytes)
            {
                let dk = data_key(key);
                let data_opt = self.storage_get(key, &dk, start_ts)?;
                if let Some(data_bytes) = data_opt {
                    if let Ok(value_opt) = bincode::deserialize::<Option<Vec<u8>>>(&data_bytes) {
                        return Ok(value_opt);
                    }
                }
            }
        }
        Ok(None)
    }

    pub fn read_sync(&self, key: &[u8], as_of: HlcTimestamp) -> Result<Option<Vec<u8>>, TxnError> {
        tokio::task::block_in_place(|| futures::executor::block_on(self.get(key, as_of)))
    }

    pub async fn prewrite_async(
        &self,
        txn_ts: HlcTimestamp,
        mutations: &[Mutation],
    ) -> Result<(), TxnError> {
        if mutations.is_empty() {
            return Ok(());
        }

        {
            let mut active = self.active_txns.lock();
            let keys: Vec<Vec<u8>> = mutations
                .iter()
                .map(|m| match m {
                    Mutation::Put(k, _) => k.clone(),
                    Mutation::Delete(k) => k.clone(),
                })
                .collect();
            active.insert(txn_ts, keys);
        }

        let primary_mutation = &mutations[0];
        let primary_key = match primary_mutation {
            Mutation::Put(k, _) => k,
            Mutation::Delete(k) => k,
        };

        // Prewrite primary first
        self.prewrite_mutation(txn_ts, primary_mutation, primary_key.clone(), 1000)
            .await?;

        // Prewrite secondaries
        for mutation in &mutations[1..] {
            if let Err(e) = self
                .prewrite_mutation(txn_ts, mutation, primary_key.clone(), 1000)
                .await
            {
                let _ = self.abort_async(txn_ts, mutations).await;
                return Err(e);
            }
        }

        Ok(())
    }

    async fn prewrite_mutation(
        &self,
        txn_ts: HlcTimestamp,
        mutation: &Mutation,
        primary: Vec<u8>,
        ttl: u64,
    ) -> Result<(), TxnError> {
        let (key, value) = match mutation {
            Mutation::Put(k, v) => (k.clone(), Some(v.clone())),
            Mutation::Delete(k) => (k.clone(), None),
        };

        let cmd = ShardCommand::TxnPrewrite {
            key: key.clone(),
            value,
            primary,
            start_ts: txn_ts,
            ttl,
        };

        self.propose_command(&key, cmd).await?;

        // Verify prewrite status
        let lk = lock_key(&key);
        let max_ts = HlcTimestamp {
            physical: u64::MAX,
            logical: u32::MAX,
        };

        let lock_opt = self.storage_get(&key, &lk, max_ts)?;
        if let Some(lock_bytes) = lock_opt {
            if let Ok(lock_info) = bincode::deserialize::<LockInfo>(&lock_bytes) {
                if lock_info.ts == txn_ts {
                    return Ok(());
                }
            }
        }

        let ek = error_key(&key, txn_ts);
        let err_opt = self.storage_get(&key, &ek, max_ts)?;
        if let Some(err_bytes) = err_opt {
            if let Ok(err) = bincode::deserialize::<PrewriteError>(&err_bytes) {
                return match err {
                    PrewriteError::WriteConflict(ts) => Err(TxnError::WriteConflict {
                        key,
                        conflict_ts: ts,
                    }),
                    PrewriteError::LockConflict { primary, ts } => Err(TxnError::LockConflict {
                        key,
                        primary,
                        lock_ts: ts,
                    }),
                };
            }
        }

        Err(TxnError::Aborted)
    }

    pub async fn commit_async(
        &self,
        txn_ts: HlcTimestamp,
        commit_ts: HlcTimestamp,
    ) -> Result<(), TxnError> {
        let keys = {
            let guard = self.active_txns.lock();
            guard.get(&txn_ts).cloned()
        };

        let keys = match keys {
            Some(k) if !k.is_empty() => k,
            _ => {
                return Err(TxnError::Other(
                    "Transaction not found or has no mutations".to_string(),
                ))
            }
        };

        let primary_key = &keys[0];

        let primary_cmd = ShardCommand::TxnCommit {
            key: primary_key.clone(),
            start_ts: txn_ts,
            commit_ts,
            is_primary: true,
        };

        self.propose_command(primary_key, primary_cmd).await?;

        // Verify primary committed successfully
        let ck = commit_key(primary_key, txn_ts);
        let max_ts = HlcTimestamp {
            physical: u64::MAX,
            logical: u32::MAX,
        };
        let commit_opt = self.storage_get(primary_key, &ck, max_ts)?;
        if commit_opt.is_none() {
            return Err(TxnError::Aborted);
        }

        // Commit secondaries concurrently
        let mut futures = Vec::new();
        for key in &keys[1..] {
            let secondary_cmd = ShardCommand::TxnCommit {
                key: key.clone(),
                start_ts: txn_ts,
                commit_ts,
                is_primary: false,
            };
            futures.push(self.propose_command(key, secondary_cmd));
        }
        let _ = futures::future::join_all(futures).await;

        self.active_txns.lock().remove(&txn_ts);
        Ok(())
    }

    pub async fn abort_async(
        &self,
        txn_ts: HlcTimestamp,
        mutations: &[Mutation],
    ) -> Result<(), TxnError> {
        let keys = {
            let mut guard = self.active_txns.lock();
            if let Some(k) = guard.remove(&txn_ts) {
                k
            } else {
                mutations
                    .iter()
                    .map(|m| match m {
                        Mutation::Put(k, _) => k.clone(),
                        Mutation::Delete(k) => k.clone(),
                    })
                    .collect()
            }
        };

        for key in keys {
            let cmd = ShardCommand::TxnRollback {
                key: key.clone(),
                start_ts: txn_ts,
            };
            let _ = self.propose_command(&key, cmd).await;
        }
        Ok(())
    }
}

impl<T: RaftTransport + 'static> TransactionCoordinator for DistributedTxnCoordinator<T> {
    fn begin(&self) -> HlcTimestamp {
        self.hlc.local_event()
    }

    fn prewrite(&self, txn_ts: HlcTimestamp, mutations: &[Mutation]) -> Result<(), TxnError> {
        {
            let mut active = self.active_txns.lock();
            let keys: Vec<Vec<u8>> = mutations
                .iter()
                .map(|m| match m {
                    Mutation::Put(k, _) => k.clone(),
                    Mutation::Delete(k) => k.clone(),
                })
                .collect();
            active.insert(txn_ts, keys);
        }

        tokio::task::block_in_place(|| {
            futures::executor::block_on(self.prewrite_async(txn_ts, mutations))
        })
    }

    fn commit(&self, txn_ts: HlcTimestamp, commit_ts: HlcTimestamp) -> Result<(), TxnError> {
        tokio::task::block_in_place(|| {
            futures::executor::block_on(self.commit_async(txn_ts, commit_ts))
        })
    }

    fn abort(&self, txn_ts: HlcTimestamp) -> Result<(), TxnError> {
        tokio::task::block_in_place(|| futures::executor::block_on(self.abort_async(txn_ts, &[])))
    }
}
