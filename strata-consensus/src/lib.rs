use parking_lot::Mutex;
use rand::SeedableRng;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use strata_storage::wal::Wal;

pub type NodeId = u64;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RequestVoteReq {
    pub shard_id: u32,
    pub term: u64,
    pub candidate_id: NodeId,
    pub last_log_index: u64,
    pub last_log_term: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RequestVoteResp {
    pub term: u64,
    pub vote_granted: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AppendEntriesReq {
    pub shard_id: u32,
    pub term: u64,
    pub leader_id: NodeId,
    pub prev_log_index: u64,
    pub prev_log_term: u64,
    pub entries: Vec<LogEntry>,
    pub leader_commit: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogEntry {
    pub term: u64,
    pub index: u64,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AppendEntriesResp {
    pub term: u64,
    pub success: bool,
    pub match_index: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InstallSnapshotReq {
    pub shard_id: u32,
    pub term: u64,
    pub leader_id: NodeId,
    pub last_included_index: u64,
    pub last_included_term: u64,
    pub offset: u64,
    pub data: Vec<u8>,
    pub done: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InstallSnapshotResp {
    pub term: u64,
}

#[derive(Debug)]
pub enum TransportError {
    ConnectionRefused,
    Timeout,
    Serialization(String),
    Other(String),
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransportError::ConnectionRefused => write!(f, "Connection refused"),
            TransportError::Timeout => write!(f, "Timeout"),
            TransportError::Serialization(e) => write!(f, "Serialization error: {}", e),
            TransportError::Other(e) => write!(f, "Transport error: {}", e),
        }
    }
}

impl std::error::Error for TransportError {}

#[derive(Debug)]
pub enum StateMachineError {
    ApplyFailed(String),
    SnapshotFailed(String),
    RestoreFailed(String),
}

impl fmt::Display for StateMachineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StateMachineError::ApplyFailed(e) => write!(f, "Apply failed: {}", e),
            StateMachineError::SnapshotFailed(e) => write!(f, "Snapshot failed: {}", e),
            StateMachineError::RestoreFailed(e) => write!(f, "Restore failed: {}", e),
        }
    }
}

impl std::error::Error for StateMachineError {}

pub trait RaftTransport: Send + Sync {
    fn send_request_vote(
        &self,
        to: NodeId,
        req: RequestVoteReq,
    ) -> Result<RequestVoteResp, TransportError>;
    fn send_append_entries(
        &self,
        to: NodeId,
        req: AppendEntriesReq,
    ) -> Result<AppendEntriesResp, TransportError>;
    fn send_install_snapshot(
        &self,
        to: NodeId,
        req: InstallSnapshotReq,
    ) -> Result<InstallSnapshotResp, TransportError>;
}

pub trait StateMachine: Send + Sync {
    fn apply(&self, command: &[u8]) -> Result<Vec<u8>, StateMachineError>;
    fn snapshot(&self) -> Result<Vec<u8>, StateMachineError>;
    fn restore(&self, snapshot: &[u8]) -> Result<(), StateMachineError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Follower,
    Candidate,
    Leader,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum ConfigState {
    Stable(Vec<NodeId>),
    Joint { old: Vec<NodeId>, new: Vec<NodeId> },
}

impl ConfigState {
    pub fn all_nodes(&self) -> Vec<NodeId> {
        match self {
            ConfigState::Stable(nodes) => nodes.clone(),
            ConfigState::Joint { old, new } => {
                let mut set: HashSet<NodeId> = old.iter().cloned().collect();
                set.extend(new.iter().cloned());
                let mut list: Vec<NodeId> = set.into_iter().collect();
                list.sort();
                list
            }
        }
    }

    pub fn is_quorum(&self, votes: &HashSet<NodeId>, self_id: NodeId) -> bool {
        match self {
            ConfigState::Stable(nodes) => {
                let count = nodes
                    .iter()
                    .filter(|&&n| votes.contains(&n) || n == self_id)
                    .count();
                count > nodes.len() / 2
            }
            ConfigState::Joint { old, new } => {
                let count_old = old
                    .iter()
                    .filter(|&&n| votes.contains(&n) || n == self_id)
                    .count();
                let count_new = new
                    .iter()
                    .filter(|&&n| votes.contains(&n) || n == self_id)
                    .count();
                count_old > old.len() / 2 && count_new > new.len() / 2
            }
        }
    }

    pub fn can_commit(
        &self,
        match_index: &HashMap<NodeId, u64>,
        self_id: NodeId,
        last_log_idx: u64,
        index: u64,
    ) -> bool {
        let get_match = |node: NodeId| {
            if node == self_id {
                last_log_idx
            } else {
                match_index.get(&node).cloned().unwrap_or(0)
            }
        };
        match self {
            ConfigState::Stable(nodes) => {
                let count = nodes.iter().filter(|&&n| get_match(n) >= index).count();
                count > nodes.len() / 2
            }
            ConfigState::Joint { old, new } => {
                let count_old = old.iter().filter(|&&n| get_match(n) >= index).count();
                let count_new = new.iter().filter(|&&n| get_match(n) >= index).count();
                count_old > old.len() / 2 && count_new > new.len() / 2
            }
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum EntryPayload {
    Command(Vec<u8>),
    ConfigJoint { old: Vec<NodeId>, new: Vec<NodeId> },
    ConfigStable(Vec<NodeId>),
}

pub struct RaftState {
    pub current_term: u64,
    pub voted_for: Option<NodeId>,
    pub log: Vec<LogEntry>,
    pub last_snapshot_index: u64,
    pub last_snapshot_term: u64,
    pub last_snapshot_data: Vec<u8>,
    pub config: ConfigState,
    pub snapshot_config: ConfigState,
    pub commit_index: u64,
    pub last_applied: u64,
    pub role: Role,
    pub next_index: HashMap<NodeId, u64>,
    pub match_index: HashMap<NodeId, u64>,
    pub votes_received: HashSet<NodeId>,
    pub election_timeout: u64,
    pub election_elapsed: u64,
    pub heartbeat_timeout: u64,
    pub heartbeat_elapsed: u64,
}

impl RaftState {
    pub fn new(id: NodeId, peers: &[NodeId]) -> Self {
        let mut next_index = HashMap::new();
        let mut match_index = HashMap::new();
        for &peer in peers {
            next_index.insert(peer, 1);
            match_index.insert(peer, 0);
        }
        let mut nodes = peers.to_vec();
        nodes.push(id);
        nodes.sort();
        Self {
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            last_snapshot_index: 0,
            last_snapshot_term: 0,
            last_snapshot_data: Vec::new(),
            config: ConfigState::Stable(nodes.clone()),
            snapshot_config: ConfigState::Stable(nodes),
            commit_index: 0,
            last_applied: 0,
            role: Role::Follower,
            next_index,
            match_index,
            votes_received: HashSet::new(),
            election_timeout: 15,
            election_elapsed: 0,
            heartbeat_timeout: 5,
            heartbeat_elapsed: 0,
        }
    }

    pub fn get_entry(&self, index: u64) -> Option<&LogEntry> {
        if index <= self.last_snapshot_index {
            None
        } else {
            let pos = (index - self.last_snapshot_index - 1) as usize;
            self.log.get(pos)
        }
    }

    pub fn get_term(&self, index: u64) -> u64 {
        if index == 0 {
            0
        } else if index == self.last_snapshot_index {
            self.last_snapshot_term
        } else {
            self.get_entry(index).map(|e| e.term).unwrap_or(0)
        }
    }

    pub fn last_log_index(&self) -> u64 {
        if self.log.is_empty() {
            self.last_snapshot_index
        } else {
            self.log.last().unwrap().index
        }
    }

    pub fn last_log_term(&self) -> u64 {
        let last_idx = self.last_log_index();
        self.get_term(last_idx)
    }

    pub fn update_active_config(&mut self) {
        let mut active = self.snapshot_config.clone();
        for entry in &self.log {
            if let Ok(payload) = bincode::deserialize::<EntryPayload>(&entry.data) {
                match payload {
                    EntryPayload::ConfigJoint { old, new } => {
                        active = ConfigState::Joint { old, new };
                    }
                    EntryPayload::ConfigStable(nodes) => {
                        active = ConfigState::Stable(nodes);
                    }
                    _ => {}
                }
            }
        }
        self.config = active;
    }

    pub fn config_at(&self, index: u64) -> ConfigState {
        let mut active = self.snapshot_config.clone();
        for entry in &self.log {
            if entry.index > index {
                break;
            }
            if let Ok(payload) = bincode::deserialize::<EntryPayload>(&entry.data) {
                match payload {
                    EntryPayload::ConfigJoint { old, new } => {
                        active = ConfigState::Joint { old, new };
                    }
                    EntryPayload::ConfigStable(nodes) => {
                        active = ConfigState::Stable(nodes);
                    }
                    _ => {}
                }
            }
        }
        active
    }
}

#[derive(Debug)]
pub enum Event {
    Tick,
    RequestVote {
        req: RequestVoteReq,
        tx: tokio::sync::oneshot::Sender<RequestVoteResp>,
    },
    AppendEntries {
        req: AppendEntriesReq,
        tx: tokio::sync::oneshot::Sender<AppendEntriesResp>,
    },
    InstallSnapshot {
        req: InstallSnapshotReq,
        tx: tokio::sync::oneshot::Sender<InstallSnapshotResp>,
    },
    RequestVoteResponse {
        peer: NodeId,
        term: u64,
        resp: RequestVoteResp,
    },
    AppendEntriesResponse {
        peer: NodeId,
        term: u64,
        resp: AppendEntriesResp,
        sent_prev_index: u64,
        sent_count: usize,
    },
    InstallSnapshotResponse {
        peer: NodeId,
        term: u64,
        resp: InstallSnapshotResp,
    },
    Propose {
        data: Vec<u8>,
        tx: tokio::sync::oneshot::Sender<Result<u64, String>>,
    },
    ProposeConfigChange {
        new_nodes: Vec<NodeId>,
        tx: tokio::sync::oneshot::Sender<Result<(), String>>,
    },
    ProposeConfigStable {
        new_nodes: Vec<NodeId>,
        tx: tokio::sync::oneshot::Sender<Result<(), String>>,
    },
    Shutdown,
}

pub struct RaftNode<S, T> {
    pub shard_id: u32,
    pub id: NodeId,
    pub peers: Vec<NodeId>,
    pub wal_path: PathBuf,
    pub state: Arc<Mutex<RaftState>>,
    pub wal: Arc<Mutex<Wal>>,
    pub state_machine: Arc<S>,
    pub transport: Arc<T>,
    pub event_tx: tokio::sync::mpsc::UnboundedSender<Event>,
    pub rng: Arc<Mutex<rand::rngs::StdRng>>,
}

pub struct HandleEventCtx<'a, S, T> {
    pub shard_id: u32,
    pub id: NodeId,
    pub peers: &'a [NodeId],
    pub wal_path: &'a Path,
    pub state_lock: &'a Arc<Mutex<RaftState>>,
    pub wal_lock: &'a Arc<Mutex<Wal>>,
    pub rng_lock: &'a Arc<Mutex<rand::rngs::StdRng>>,
    pub state_machine: &'a Arc<S>,
    pub transport: &'a Arc<T>,
    pub event_tx: &'a tokio::sync::mpsc::UnboundedSender<Event>,
}

impl<S: StateMachine + 'static, T: RaftTransport + 'static> RaftNode<S, T> {
    pub fn new(
        shard_id: u32,
        id: NodeId,
        peers: Vec<NodeId>,
        wal_path: PathBuf,
        state_machine: Arc<S>,
        transport: Arc<T>,
    ) -> Result<Self, std::io::Error> {
        if let Some(parent) = wal_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let snapshot_path = wal_path.with_file_name(format!("snapshot_node_{}.bin", id));
        let mut default_nodes = peers.clone();
        default_nodes.push(id);
        default_nodes.sort();

        let (last_included_index, last_included_term, snapshot_config, snapshot_payload) =
            if snapshot_path.exists() {
                let data = fs::read(&snapshot_path)?;
                if data.len() >= 20 {
                    let last_idx = u64::from_le_bytes(data[0..8].try_into().unwrap());
                    let last_term = u64::from_le_bytes(data[8..16].try_into().unwrap());
                    let config_len = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
                    let config: ConfigState =
                        bincode::deserialize(&data[20..20 + config_len]).unwrap();
                    let payload = data[20 + config_len..].to_vec();
                    (last_idx, last_term, config, payload)
                } else {
                    (0, 0, ConfigState::Stable(default_nodes.clone()), Vec::new())
                }
            } else {
                (0, 0, ConfigState::Stable(default_nodes.clone()), Vec::new())
            };

        if last_included_index > 0 {
            state_machine
                .restore(&snapshot_payload)
                .map_err(|e| io::Error::other(format!("Restore failed: {}", e)))?;
        }

        let mut wal = Wal::new(&wal_path)?;
        let mut current_term = 0;
        let mut voted_for = None;
        let mut log_entries = HashMap::new();
        let mut log_len = 0u64;

        wal.replay(|is_delete, key, value, _ts| {
            if key == b"term" {
                if !is_delete {
                    if let Ok(arr) = value.try_into() {
                        current_term = u64::from_le_bytes(arr);
                    }
                }
            } else if key == b"vote" {
                if !is_delete {
                    if let Ok(v) = bincode::deserialize(&value) {
                        voted_for = v;
                    }
                } else {
                    voted_for = None;
                }
            } else if key == b"log_len" {
                if !is_delete {
                    if let Ok(arr) = value.try_into() {
                        log_len = u64::from_le_bytes(arr);
                    }
                }
            } else if key.starts_with(b"entry_") {
                if let Ok(index_str) = std::str::from_utf8(&key[6..]) {
                    if let Ok(index) = index_str.parse::<u64>() {
                        if !is_delete {
                            if let Ok(entry) = bincode::deserialize::<LogEntry>(&value) {
                                log_entries.insert(index, entry);
                            }
                        } else {
                            log_entries.remove(&index);
                        }
                    }
                }
            }
        })?;

        let mut state = RaftState::new(id, &peers);
        state.current_term = current_term;
        state.voted_for = voted_for;
        state.last_snapshot_index = last_included_index;
        state.last_snapshot_term = last_included_term;
        state.snapshot_config = snapshot_config;
        state.last_snapshot_data = snapshot_payload;
        state.commit_index = last_included_index;
        state.last_applied = last_included_index;

        let mut log = Vec::new();
        for idx in (last_included_index + 1)..=log_len {
            if let Some(entry) = log_entries.remove(&idx) {
                log.push(entry);
            }
        }
        state.log = log;
        state.update_active_config();

        for idx in (state.last_applied + 1)..=state.commit_index {
            if let Some(entry) = state.get_entry(idx) {
                if let Ok(EntryPayload::Command(cmd)) =
                    bincode::deserialize::<EntryPayload>(&entry.data)
                {
                    let _ = state_machine.apply(&cmd);
                }
            }
        }
        state.last_applied = state.commit_index;

        let mut rng = rand::rngs::StdRng::seed_from_u64(id);
        use rand::Rng;
        state.election_timeout = rng.gen_range(15..30);
        state.election_elapsed = 0;
        state.heartbeat_timeout = 5;
        state.heartbeat_elapsed = 0;

        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let state_arc = Arc::new(Mutex::new(state));
        let wal_arc = Arc::new(Mutex::new(wal));
        let rng_arc = Arc::new(Mutex::new(rng));

        let loop_id = id;
        let loop_peers = peers.clone();
        let loop_wal_path = wal_path.clone();
        let loop_state = state_arc.clone();
        let loop_wal = wal_arc.clone();
        let loop_rng = rng_arc.clone();
        let loop_sm = state_machine.clone();
        let loop_transport = transport.clone();
        let loop_tx = event_tx.clone();

        tokio::spawn(async move {
            while let Some(event) = event_rx.recv().await {
                if matches!(event, Event::Shutdown) {
                    break;
                }
                let mut ctx = HandleEventCtx {
                    shard_id,
                    id: loop_id,
                    peers: &loop_peers,
                    wal_path: &loop_wal_path,
                    state_lock: &loop_state,
                    wal_lock: &loop_wal,
                    rng_lock: &loop_rng,
                    state_machine: &loop_sm,
                    transport: &loop_transport,
                    event_tx: &loop_tx,
                };
                Self::handle_event(&mut ctx, event);
            }
        });

        Ok(Self {
            shard_id,
            id,
            peers,
            wal_path,
            state: state_arc,
            wal: wal_arc,
            state_machine,
            transport,
            event_tx,
            rng: rng_arc,
        })
    }

    pub fn shutdown(&self) {
        let _ = self.event_tx.send(Event::Shutdown);
    }

    pub fn propose(&self, data: Vec<u8>) -> tokio::sync::oneshot::Receiver<Result<u64, String>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.event_tx.send(Event::Propose { data, tx });
        rx
    }

    pub fn change_membership(
        &self,
        new_nodes: Vec<NodeId>,
    ) -> tokio::sync::oneshot::Receiver<Result<(), String>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self
            .event_tx
            .send(Event::ProposeConfigChange { new_nodes, tx });
        rx
    }

    pub fn tick(&self) {
        let _ = self.event_tx.send(Event::Tick);
    }

    pub async fn handle_request_vote_rpc(&self, req: RequestVoteReq) -> RequestVoteResp {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.event_tx.send(Event::RequestVote { req, tx });
        rx.await.unwrap()
    }

    pub async fn handle_append_entries_rpc(&self, req: AppendEntriesReq) -> AppendEntriesResp {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.event_tx.send(Event::AppendEntries { req, tx });
        rx.await.unwrap()
    }

    pub async fn handle_install_snapshot_rpc(
        &self,
        req: InstallSnapshotReq,
    ) -> InstallSnapshotResp {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.event_tx.send(Event::InstallSnapshot { req, tx });
        rx.await.unwrap()
    }

    pub fn take_snapshot(&self) -> Result<(), String> {
        let mut state = self.state.lock();
        if state.last_applied <= state.last_snapshot_index {
            return Ok(());
        }

        let snapshot_data = self.state_machine.snapshot().map_err(|e| e.to_string())?;
        let last_included_index = state.last_applied;
        let last_included_term = state.get_term(last_included_index);

        state.last_snapshot_index = last_included_index;
        state.last_snapshot_term = last_included_term;
        state.last_snapshot_data = snapshot_data.clone();

        // Use committed config as snapshot configuration
        state.snapshot_config = state.config_at(last_included_index);

        let mut new_log = Vec::new();
        for idx in (state.last_snapshot_index + 1)..=state.last_log_index() {
            if let Some(entry) = state.get_entry(idx) {
                new_log.push(entry.clone());
            }
        }
        state.log = new_log;

        let snapshot_path = self
            .wal_path
            .with_file_name(format!("snapshot_node_{}.bin", self.id));
        let config_bytes = bincode::serialize(&state.snapshot_config).unwrap();
        let mut data = Vec::with_capacity(20 + config_bytes.len() + snapshot_data.len());
        data.extend_from_slice(&last_included_index.to_le_bytes());
        data.extend_from_slice(&last_included_term.to_le_bytes());
        data.extend_from_slice(&(config_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(&config_bytes);
        data.extend_from_slice(&snapshot_data);
        std::fs::write(&snapshot_path, data).map_err(|e| e.to_string())?;

        Self::rewrite_wal(&self.wal_path, self.id, &state, &self.wal).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn handle_event(ctx: &mut HandleEventCtx<'_, S, T>, event: Event) {
        match event {
            Event::Tick => {
                let mut state = ctx.state_lock.lock();
                if state.role == Role::Leader {
                    state.heartbeat_elapsed += 1;
                    if state.heartbeat_elapsed >= state.heartbeat_timeout {
                        state.heartbeat_elapsed = 0;
                        Self::broadcast_append_entries(
                            ctx.shard_id,
                            ctx.id,
                            ctx.peers,
                            &state,
                            ctx.transport,
                            ctx.event_tx,
                        );
                    }
                } else {
                    state.election_elapsed += 1;
                    if state.election_elapsed >= state.election_timeout {
                        state.election_elapsed = 0;
                        state.role = Role::Candidate;
                        state.current_term += 1;
                        state.voted_for = Some(ctx.id);
                        let mut rng = ctx.rng_lock.lock();
                        use rand::Rng;
                        state.election_timeout = rng.gen_range(15..30);

                        let _ = Self::persist_term_and_vote(
                            ctx.wal_lock,
                            state.current_term,
                            state.voted_for,
                        );

                        state.votes_received.clear();
                        state.votes_received.insert(ctx.id);

                        if state.config.is_quorum(&state.votes_received, ctx.id) {
                            state.role = Role::Leader;
                            let last_idx = state.last_log_index();
                            let all = state.config.all_nodes();
                            for &peer in &all {
                                if peer != ctx.id {
                                    state.next_index.insert(peer, last_idx + 1);
                                    state.match_index.insert(peer, 0);
                                }
                            }
                            Self::broadcast_append_entries(
                                ctx.shard_id,
                                ctx.id,
                                ctx.peers,
                                &state,
                                ctx.transport,
                                ctx.event_tx,
                            );
                        } else {
                            let term = state.current_term;
                            let last_log_idx = state.last_log_index();
                            let last_log_t = state.last_log_term();
                            let all = state.config.all_nodes();
                            for &peer in &all {
                                if peer != ctx.id {
                                    let tx = ctx.event_tx.clone();
                                    let transport_clone = ctx.transport.clone();
                                    let req = RequestVoteReq {
                                        shard_id: ctx.shard_id,
                                        term,
                                        candidate_id: ctx.id,
                                        last_log_index: last_log_idx,
                                        last_log_term: last_log_t,
                                    };
                                    std::thread::spawn(move || {
                                        if let Ok(resp) =
                                            transport_clone.send_request_vote(peer, req)
                                        {
                                            let _ = tx.send(Event::RequestVoteResponse {
                                                peer,
                                                term,
                                                resp,
                                            });
                                        }
                                    });
                                }
                            }
                        }
                    }
                }
            }
            Event::RequestVote { req, tx } => {
                let mut state = ctx.state_lock.lock();
                let mut stepped_down = false;
                if req.term > state.current_term {
                    state.current_term = req.term;
                    state.voted_for = None;
                    state.role = Role::Follower;
                    stepped_down = true;
                }

                let mut vote_granted = false;
                if req.term == state.current_term
                    && (state.voted_for.is_none() || state.voted_for == Some(req.candidate_id))
                {
                    let last_log_idx = state.last_log_index();
                    let last_log_t = state.last_log_term();
                    let log_ok = req.last_log_term > last_log_t
                        || (req.last_log_term == last_log_t && req.last_log_index >= last_log_idx);
                    if log_ok {
                        vote_granted = true;
                        state.voted_for = Some(req.candidate_id);
                        state.election_elapsed = 0;
                        stepped_down = true;
                    }
                }

                if stepped_down {
                    let _ = Self::persist_term_and_vote(
                        ctx.wal_lock,
                        state.current_term,
                        state.voted_for,
                    );
                }

                let _ = tx.send(RequestVoteResp {
                    term: state.current_term,
                    vote_granted,
                });
            }
            Event::AppendEntries { req, tx } => {
                let mut state = ctx.state_lock.lock();
                let mut stepped_down = false;
                if req.term > state.current_term {
                    state.current_term = req.term;
                    state.voted_for = None;
                    state.role = Role::Follower;
                    stepped_down = true;
                }

                if req.term < state.current_term {
                    let _ = tx.send(AppendEntriesResp {
                        term: state.current_term,
                        success: false,
                        match_index: state.last_log_index(),
                    });
                    if stepped_down {
                        let _ = Self::persist_term_and_vote(
                            ctx.wal_lock,
                            state.current_term,
                            state.voted_for,
                        );
                    }
                    return;
                }

                if state.role == Role::Candidate {
                    state.role = Role::Follower;
                    stepped_down = true;
                }
                state.election_elapsed = 0;

                let last_idx = state.last_log_index();
                let log_match = if req.prev_log_index > last_idx {
                    false
                } else if req.prev_log_index < state.last_snapshot_index {
                    true
                } else {
                    state.get_term(req.prev_log_index) == req.prev_log_term
                };

                if !log_match {
                    let _ = tx.send(AppendEntriesResp {
                        term: state.current_term,
                        success: false,
                        match_index: state.last_log_index(),
                    });
                    if stepped_down {
                        let _ = Self::persist_term_and_vote(
                            ctx.wal_lock,
                            state.current_term,
                            state.voted_for,
                        );
                    }
                    return;
                }

                let mut config_changed = false;
                for entry in req.entries {
                    let idx = entry.index;
                    let existing_term = state.get_term(idx);
                    if existing_term != 0 {
                        if existing_term != entry.term {
                            let pos = (idx - state.last_snapshot_index - 1) as usize;
                            state.log.truncate(pos);
                            let _ = Self::persist_log_len(ctx.wal_lock, state.last_log_index());
                            state.log.push(entry.clone());
                            let _ = Self::persist_log_entry(ctx.wal_lock, &entry);
                            config_changed = true;
                        }
                    } else {
                        state.log.push(entry.clone());
                        let _ = Self::persist_log_entry(ctx.wal_lock, &entry);
                        config_changed = true;
                    }
                }

                if config_changed {
                    state.update_active_config();
                }

                let mut apply_needed = false;
                if req.leader_commit > state.commit_index {
                    state.commit_index = req.leader_commit.min(state.last_log_index());
                    apply_needed = true;
                }

                if stepped_down {
                    let _ = Self::persist_term_and_vote(
                        ctx.wal_lock,
                        state.current_term,
                        state.voted_for,
                    );
                }

                let term = state.current_term;
                let last_log_idx = state.last_log_index();
                drop(state);

                if apply_needed {
                    Self::apply_committed(ctx.state_lock, ctx.state_machine);
                }

                let _ = tx.send(AppendEntriesResp {
                    term,
                    success: true,
                    match_index: last_log_idx,
                });
            }
            Event::InstallSnapshot { req, tx } => {
                let mut state = ctx.state_lock.lock();
                let mut stepped_down = false;
                if req.term > state.current_term {
                    state.current_term = req.term;
                    state.voted_for = None;
                    state.role = Role::Follower;
                    stepped_down = true;
                }

                if req.term < state.current_term {
                    let _ = tx.send(InstallSnapshotResp {
                        term: state.current_term,
                    });
                    if stepped_down {
                        let _ = Self::persist_term_and_vote(
                            ctx.wal_lock,
                            state.current_term,
                            state.voted_for,
                        );
                    }
                    return;
                }

                if state.role == Role::Candidate {
                    state.role = Role::Follower;
                    stepped_down = true;
                }
                state.election_elapsed = 0;

                // Load snapshot header
                if req.last_included_index > state.last_snapshot_index {
                    // Extract snapshot payload
                    if let Ok(()) = ctx.state_machine.restore(&req.data) {
                        // The InstallSnapshot request carries payload directly
                        // We also set the configuration
                        state.last_snapshot_index = req.last_included_index;
                        state.last_snapshot_term = req.last_included_term;
                        state.last_snapshot_data = req.data.clone();
                        // Recover snapshot config if it fits or we just keep it
                        state.snapshot_config = ConfigState::Stable(ctx.peers.to_vec()); // Default stable
                        state.commit_index = state.commit_index.max(req.last_included_index);
                        state.last_applied = state.last_applied.max(req.last_included_index);

                        let mut new_log = Vec::new();
                        for idx in (state.last_snapshot_index + 1)..=state.last_log_index() {
                            if let Some(entry) = state.get_entry(idx) {
                                new_log.push(entry.clone());
                            }
                        }
                        state.log = new_log;
                        state.update_active_config();

                        let snapshot_path = ctx
                            .wal_path
                            .with_file_name(format!("snapshot_node_{}.bin", ctx.id));
                        let config_bytes = bincode::serialize(&state.snapshot_config).unwrap();
                        let mut snapshot_data =
                            Vec::with_capacity(20 + config_bytes.len() + req.data.len());
                        snapshot_data.extend_from_slice(&req.last_included_index.to_le_bytes());
                        snapshot_data.extend_from_slice(&req.last_included_term.to_le_bytes());
                        snapshot_data.extend_from_slice(&(config_bytes.len() as u32).to_le_bytes());
                        snapshot_data.extend_from_slice(&config_bytes);
                        snapshot_data.extend_from_slice(&req.data);
                        let _ = fs::write(&snapshot_path, snapshot_data);

                        let _ = Self::rewrite_wal(ctx.wal_path, ctx.id, &state, ctx.wal_lock);
                    }
                }

                if stepped_down {
                    let _ = Self::persist_term_and_vote(
                        ctx.wal_lock,
                        state.current_term,
                        state.voted_for,
                    );
                }

                let _ = tx.send(InstallSnapshotResp {
                    term: state.current_term,
                });
            }
            Event::RequestVoteResponse { peer, term, resp } => {
                let mut state = ctx.state_lock.lock();
                if term != state.current_term || state.role != Role::Candidate {
                    return;
                }
                if resp.term > state.current_term {
                    state.current_term = resp.term;
                    state.voted_for = None;
                    state.role = Role::Follower;
                    let _ = Self::persist_term_and_vote(
                        ctx.wal_lock,
                        state.current_term,
                        state.voted_for,
                    );
                    return;
                }
                if resp.vote_granted {
                    state.votes_received.insert(peer);
                    if state.config.is_quorum(&state.votes_received, ctx.id) {
                        state.role = Role::Leader;
                        let last_idx = state.last_log_index();
                        let all = state.config.all_nodes();
                        for &p in &all {
                            if p != ctx.id {
                                state.next_index.insert(p, last_idx + 1);
                                state.match_index.insert(p, 0);
                            }
                        }
                        state.heartbeat_elapsed = 0;
                        Self::broadcast_append_entries(
                            ctx.shard_id,
                            ctx.id,
                            ctx.peers,
                            &state,
                            ctx.transport,
                            ctx.event_tx,
                        );
                    }
                }
            }
            Event::AppendEntriesResponse {
                peer,
                term,
                resp,
                sent_prev_index,
                sent_count,
            } => {
                let mut state = ctx.state_lock.lock();
                if term != state.current_term || state.role != Role::Leader {
                    return;
                }
                if resp.term > state.current_term {
                    state.current_term = resp.term;
                    state.voted_for = None;
                    state.role = Role::Follower;
                    let _ = Self::persist_term_and_vote(
                        ctx.wal_lock,
                        state.current_term,
                        state.voted_for,
                    );
                    return;
                }
                let mut apply_needed = false;
                if resp.success {
                    let match_idx = sent_prev_index + sent_count as u64;
                    let m = state.match_index.get(&peer).cloned().unwrap_or(0);
                    if match_idx > m {
                        state.match_index.insert(peer, match_idx);
                        state.next_index.insert(peer, match_idx + 1);
                    }

                    // Check joint configuration quorums to commit
                    let last_idx = state.last_log_index();
                    let mut possible_commit = state.commit_index;
                    for idx in (state.commit_index + 1)..=last_idx {
                        if state.get_term(idx) == state.current_term
                            && state
                                .config
                                .can_commit(&state.match_index, ctx.id, last_idx, idx)
                        {
                            possible_commit = idx;
                        }
                    }
                    if possible_commit > state.commit_index {
                        state.commit_index = possible_commit;
                        apply_needed = true;
                    }
                } else {
                    let next = state.next_index.get(&peer).cloned().unwrap_or(1);
                    if next > 1 {
                        state.next_index.insert(peer, next - 1);
                        Self::send_append_entries_to_peer(
                            ctx.shard_id,
                            ctx.id,
                            peer,
                            &state,
                            ctx.transport,
                            ctx.event_tx,
                        );
                    }
                }

                drop(state);
                if apply_needed {
                    Self::apply_committed(ctx.state_lock, ctx.state_machine);
                }

                // Check if we need to propose the stable config now that the joint config is committed
                let mut stable_to_propose: Option<Vec<NodeId>> = None;
                {
                    let state_check = ctx.state_lock.lock();
                    if state_check.role == Role::Leader {
                        let committed_config = state_check.config_at(state_check.commit_index);
                        if let ConfigState::Joint { old: _, new } = committed_config {
                            if let ConfigState::Joint { .. } = state_check.config {
                                stable_to_propose = Some(new);
                            }
                        }
                    }
                }
                if let Some(new_nodes) = stable_to_propose {
                    let (tx, _rx) = tokio::sync::oneshot::channel();
                    let _ = ctx
                        .event_tx
                        .send(Event::ProposeConfigStable { new_nodes, tx });
                }
            }
            Event::InstallSnapshotResponse { peer, term, resp } => {
                let mut state = ctx.state_lock.lock();
                if term != state.current_term || state.role != Role::Leader {
                    return;
                }
                if resp.term > state.current_term {
                    state.current_term = resp.term;
                    state.voted_for = None;
                    state.role = Role::Follower;
                    let _ = Self::persist_term_and_vote(
                        ctx.wal_lock,
                        state.current_term,
                        state.voted_for,
                    );
                    return;
                }
                let last_snapshot_idx = state.last_snapshot_index;
                state.match_index.insert(peer, last_snapshot_idx);
                state.next_index.insert(peer, last_snapshot_idx + 1);
            }
            Event::Propose { data, tx } => {
                let mut state = ctx.state_lock.lock();
                if state.role != Role::Leader {
                    let _ = tx.send(Err("Not leader".to_string()));
                    return;
                }
                let index = state.last_log_index() + 1;
                let payload = EntryPayload::Command(data);
                let serialized = bincode::serialize(&payload).unwrap();
                let entry = LogEntry {
                    term: state.current_term,
                    index,
                    data: serialized,
                };
                state.log.push(entry.clone());
                let _ = Self::persist_log_entry(ctx.wal_lock, &entry);
                let _ = tx.send(Ok(index));
                Self::broadcast_append_entries(
                    ctx.shard_id,
                    ctx.id,
                    ctx.peers,
                    &state,
                    ctx.transport,
                    ctx.event_tx,
                );
            }
            Event::ProposeConfigChange { new_nodes, tx } => {
                let mut state = ctx.state_lock.lock();
                if state.role != Role::Leader {
                    let _ = tx.send(Err("Not leader".to_string()));
                    return;
                }
                match &state.config {
                    ConfigState::Stable(old) => {
                        let index = state.last_log_index() + 1;
                        let payload = EntryPayload::ConfigJoint {
                            old: old.clone(),
                            new: new_nodes.clone(),
                        };
                        let data = bincode::serialize(&payload).unwrap();
                        let entry = LogEntry {
                            term: state.current_term,
                            index,
                            data,
                        };
                        state.log.push(entry.clone());
                        let _ = Self::persist_log_entry(ctx.wal_lock, &entry);
                        state.update_active_config();

                        let last_idx = state.last_log_index();
                        for &node in state.config.all_nodes().iter() {
                            if node != ctx.id {
                                state.next_index.entry(node).or_insert(last_idx + 1);
                                state.match_index.entry(node).or_insert(0);
                            }
                        }
                        let _ = tx.send(Ok(()));
                        Self::broadcast_append_entries(
                            ctx.shard_id,
                            ctx.id,
                            ctx.peers,
                            &state,
                            ctx.transport,
                            ctx.event_tx,
                        );
                    }
                    ConfigState::Joint { .. } => {
                        let _ =
                            tx.send(Err("Configuration change already in progress".to_string()));
                    }
                }
            }
            Event::ProposeConfigStable { new_nodes, tx } => {
                let mut state = ctx.state_lock.lock();
                if state.role != Role::Leader {
                    let _ = tx.send(Err("Not leader".to_string()));
                    return;
                }
                if let ConfigState::Joint { .. } = &state.config {
                    let index = state.last_log_index() + 1;
                    let payload = EntryPayload::ConfigStable(new_nodes.clone());
                    let data = bincode::serialize(&payload).unwrap();
                    let entry = LogEntry {
                        term: state.current_term,
                        index,
                        data,
                    };
                    state.log.push(entry.clone());
                    let _ = Self::persist_log_entry(ctx.wal_lock, &entry);
                    state.update_active_config();
                    let _ = tx.send(Ok(()));
                    Self::broadcast_append_entries(
                        ctx.shard_id,
                        ctx.id,
                        ctx.peers,
                        &state,
                        ctx.transport,
                        ctx.event_tx,
                    );
                }
            }
            Event::Shutdown => {}
        }
    }

    fn apply_committed(state_lock: &Arc<Mutex<RaftState>>, state_machine: &Arc<S>) {
        let (to_apply, last_applied) = {
            let state = state_lock.lock();
            let mut to_apply = Vec::new();
            for idx in (state.last_applied + 1)..=state.commit_index {
                if let Some(entry) = state.get_entry(idx) {
                    to_apply.push((idx, entry.data.clone()));
                }
            }
            (to_apply, state.last_applied)
        };

        let mut max_applied = last_applied;
        for (idx, data) in to_apply {
            if let Ok(EntryPayload::Command(cmd)) = bincode::deserialize::<EntryPayload>(&data) {
                let _ = state_machine.apply(&cmd);
            }
            max_applied = idx;
        }

        let mut state = state_lock.lock();
        state.last_applied = max_applied;
    }

    fn broadcast_append_entries(
        shard_id: u32,
        id: NodeId,
        _peers: &[NodeId],
        state: &RaftState,
        transport: &Arc<T>,
        event_tx: &tokio::sync::mpsc::UnboundedSender<Event>,
    ) {
        let all = state.config.all_nodes();
        for &peer in &all {
            if peer != id {
                Self::send_append_entries_to_peer(shard_id, id, peer, state, transport, event_tx);
            }
        }
    }

    fn send_append_entries_to_peer(
        shard_id: u32,
        id: NodeId,
        peer: NodeId,
        state: &RaftState,
        transport: &Arc<T>,
        event_tx: &tokio::sync::mpsc::UnboundedSender<Event>,
    ) {
        let next = state.next_index.get(&peer).cloned().unwrap_or(1);
        if next <= state.last_snapshot_index {
            let req = InstallSnapshotReq {
                shard_id,
                term: state.current_term,
                leader_id: id,
                last_included_index: state.last_snapshot_index,
                last_included_term: state.last_snapshot_term,
                offset: 0,
                data: state.last_snapshot_data.clone(),
                done: true,
            };

            let tx = event_tx.clone();
            let transport_clone = transport.clone();
            let term = state.current_term;
            std::thread::spawn(move || {
                if let Ok(resp) = transport_clone.send_install_snapshot(peer, req) {
                    let _ = tx.send(Event::InstallSnapshotResponse { peer, term, resp });
                }
            });
        } else {
            let prev_log_index = next - 1;
            let prev_log_term = state.get_term(prev_log_index);
            let mut entries = Vec::new();
            for idx in next..=state.last_log_index() {
                if let Some(entry) = state.get_entry(idx) {
                    entries.push(entry.clone());
                }
            }

            let req = AppendEntriesReq {
                shard_id,
                term: state.current_term,
                leader_id: id,
                prev_log_index,
                prev_log_term,
                entries,
                leader_commit: state.commit_index,
            };

            let tx = event_tx.clone();
            let transport_clone = transport.clone();
            let term = state.current_term;
            let sent_count = req.entries.len();
            std::thread::spawn(move || {
                if let Ok(resp) = transport_clone.send_append_entries(peer, req) {
                    let _ = tx.send(Event::AppendEntriesResponse {
                        peer,
                        term,
                        resp,
                        sent_prev_index: prev_log_index,
                        sent_count,
                    });
                }
            });
        }
    }

    fn persist_term_and_vote(
        wal_lock: &Arc<Mutex<Wal>>,
        term: u64,
        vote: Option<NodeId>,
    ) -> std::io::Result<()> {
        let mut wal = wal_lock.lock();
        let ts = strata_storage::HlcTimestamp {
            physical: 0,
            logical: 0,
        };
        wal.append(false, b"term", &term.to_le_bytes(), ts)?;
        let vote_bytes = bincode::serialize(&vote).unwrap();
        wal.append(false, b"vote", &vote_bytes, ts)?;
        Ok(())
    }

    fn persist_log_entry(wal_lock: &Arc<Mutex<Wal>>, entry: &LogEntry) -> std::io::Result<()> {
        let mut wal = wal_lock.lock();
        let ts = strata_storage::HlcTimestamp {
            physical: 0,
            logical: 0,
        };
        let key = format!("entry_{:020}", entry.index).into_bytes();
        let val = bincode::serialize(entry).unwrap();
        wal.append(false, &key, &val, ts)?;
        wal.append(false, b"log_len", &entry.index.to_le_bytes(), ts)?;
        Ok(())
    }

    fn persist_log_len(wal_lock: &Arc<Mutex<Wal>>, new_len: u64) -> std::io::Result<()> {
        let mut wal = wal_lock.lock();
        let ts = strata_storage::HlcTimestamp {
            physical: 0,
            logical: 0,
        };
        wal.append(false, b"log_len", &new_len.to_le_bytes(), ts)?;
        Ok(())
    }

    fn rewrite_wal(
        wal_path: &Path,
        _id: NodeId,
        state: &RaftState,
        wal_lock: &Arc<Mutex<Wal>>,
    ) -> std::io::Result<()> {
        let mut wal_guard = wal_lock.lock();
        let _ = std::fs::remove_file(wal_path);
        let mut new_wal = Wal::new(wal_path)?;

        let ts = strata_storage::HlcTimestamp {
            physical: 0,
            logical: 0,
        };
        new_wal.append(false, b"term", &state.current_term.to_le_bytes(), ts)?;
        let vote_bytes = bincode::serialize(&state.voted_for).unwrap();
        new_wal.append(false, b"vote", &vote_bytes, ts)?;

        for entry in &state.log {
            let key = format!("entry_{:020}", entry.index).into_bytes();
            let val = bincode::serialize(entry).unwrap();
            new_wal.append(false, &key, &val, ts)?;
        }

        let last_idx = state.last_log_index();
        new_wal.append(false, b"log_len", &last_idx.to_le_bytes(), ts)?;

        *wal_guard = new_wal;
        Ok(())
    }
}
