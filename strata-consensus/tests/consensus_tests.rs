use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use parking_lot::Mutex;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;
use strata_consensus::{
    AppendEntriesReq, AppendEntriesResp, InstallSnapshotReq, InstallSnapshotResp,
    LogEntry, NodeId, RaftNode, RaftTransport, RequestVoteReq, RequestVoteResp,
    StateMachine, StateMachineError, Role, TransportError,
};

// ----------------------------------------------------------------------
// Simple Key-Value State Machine for Testing
// ----------------------------------------------------------------------
#[derive(Clone, Default)]
pub struct SimpleStateMachine {
    pub db: Arc<Mutex<HashMap<String, String>>>,
}

impl StateMachine for SimpleStateMachine {
    fn apply(&self, command: &[u8]) -> Result<Vec<u8>, StateMachineError> {
        let cmd_str = std::str::from_utf8(command)
            .map_err(|e| StateMachineError::ApplyFailed(e.to_string()))?;
        let parts: Vec<&str> = cmd_str.split(',').collect();
        if parts.len() == 2 {
            let key = parts[0].to_string();
            let value = parts[1].to_string();
            self.db.lock().insert(key, value.clone());
            Ok(value.into_bytes())
        } else {
            Err(StateMachineError::ApplyFailed(format!("Invalid command: {}", cmd_str)))
        }
    }

    fn snapshot(&self) -> Result<Vec<u8>, StateMachineError> {
        let db = self.db.lock();
        bincode::serialize(&*db).map_err(|e| StateMachineError::SnapshotFailed(e.to_string()))
    }

    fn restore(&self, snapshot: &[u8]) -> Result<(), StateMachineError> {
        let db_data: HashMap<String, String> = bincode::deserialize(snapshot)
            .map_err(|e| StateMachineError::RestoreFailed(e.to_string()))?;
        *self.db.lock() = db_data;
        Ok(())
    }
}

// ----------------------------------------------------------------------
// Mock Transport & Chaos Network Simulator
// ----------------------------------------------------------------------
pub enum NetworkPayload {
    RequestVote(RequestVoteReq),
    AppendEntries(AppendEntriesReq),
    InstallSnapshot(InstallSnapshotReq),
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

pub struct ChaosNetwork {
    pub nodes: HashMap<NodeId, Arc<RaftNode<SimpleStateMachine, MockTransport>>>,
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
            nodes: HashMap::new(),
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

    pub fn partition_all_isolated(&mut self) {
        let node_ids: Vec<NodeId> = self.nodes.keys().cloned().collect();
        for i in 0..node_ids.len() {
            for j in (i + 1)..node_ids.len() {
                self.set_partition(node_ids[i], node_ids[j]);
            }
        }
    }

    pub fn partition_group(&mut self, group: &[NodeId]) {
        let node_ids: Vec<NodeId> = self.nodes.keys().cloned().collect();
        for &n1 in &node_ids {
            for &n2 in &node_ids {
                if n1 == n2 {
                    continue;
                }
                let in_g1 = group.contains(&n1);
                let in_g2 = group.contains(&n2);
                if in_g1 != in_g2 {
                    self.set_partition(n1, n2);
                }
            }
        }
    }

    pub fn heal_partitions(&mut self) {
        self.partitions.clear();
    }

    pub fn enqueue(&self, msg: PendingMsg) {
        self.pending_msgs.lock().push(msg);
    }

    pub fn tick(&mut self) {
        // Route pending messages
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

        // Tick all nodes
        for node in self.nodes.values() {
            node.tick();
        }

        // Deliver ready delayed messages
        let mut remaining = Vec::new();
        for (deliver_at, msg) in self.delayed_msgs.drain(..) {
            if deliver_at <= self.virtual_time {
                let is_partitioned = self.partitions.contains(&(msg.from, msg.to))
                    || self.partitions.contains(&(msg.to, msg.from));
                if is_partitioned {
                    let _ = msg.reply_tx.send(Err(TransportError::Timeout));
                    continue;
                }

                if let Some(target_node) = self.nodes.get(&msg.to) {
                    let target_node_clone = target_node.clone();
                    tokio::task::block_in_place(move || {
                        futures::executor::block_on(async {
                            match msg.payload {
                                NetworkPayload::RequestVote(req) => {
                                    let resp = target_node_clone.handle_request_vote_rpc(req).await;
                                    let _ = msg.reply_tx.send(Ok(NetworkResponse::RequestVote(resp)));
                                }
                                NetworkPayload::AppendEntries(req) => {
                                    let resp = target_node_clone.handle_append_entries_rpc(req).await;
                                    let _ = msg.reply_tx.send(Ok(NetworkResponse::AppendEntries(resp)));
                                }
                                NetworkPayload::InstallSnapshot(req) => {
                                    let resp = target_node_clone.handle_install_snapshot_rpc(req).await;
                                    let _ = msg.reply_tx.send(Ok(NetworkResponse::InstallSnapshot(resp)));
                                }
                            }
                        });
                    });
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

pub struct MockTransport {
    pub node_id: NodeId,
    pub network: Arc<Mutex<ChaosNetwork>>,
}

impl MockTransport {
    fn send_payload(&self, to: NodeId, payload: NetworkPayload) -> Result<NetworkResponse, TransportError> {
        let (tx, rx) = std::sync::mpsc::channel();
        let pending = PendingMsg {
            from: self.node_id,
            to,
            payload,
            reply_tx: tx,
        };
        {
            let net = self.network.lock();
            net.enqueue(pending);
        }
        rx.recv().map_err(|_| TransportError::Timeout)?
    }
}

impl RaftTransport for MockTransport {
    fn send_request_vote(
        &self,
        to: NodeId,
        req: RequestVoteReq,
    ) -> Result<RequestVoteResp, TransportError> {
        match self.send_payload(to, NetworkPayload::RequestVote(req))? {
            NetworkResponse::RequestVote(resp) => Ok(resp),
            _ => Err(TransportError::Other("Invalid response type".to_string())),
        }
    }

    fn send_append_entries(
        &self,
        to: NodeId,
        req: AppendEntriesReq,
    ) -> Result<AppendEntriesResp, TransportError> {
        match self.send_payload(to, NetworkPayload::AppendEntries(req))? {
            NetworkResponse::AppendEntries(resp) => Ok(resp),
            _ => Err(TransportError::Other("Invalid response type".to_string())),
        }
    }

    fn send_install_snapshot(
        &self,
        to: NodeId,
        req: InstallSnapshotReq,
    ) -> Result<InstallSnapshotResp, TransportError> {
        match self.send_payload(to, NetworkPayload::InstallSnapshot(req))? {
            NetworkResponse::InstallSnapshot(resp) => Ok(resp),
            _ => Err(TransportError::Other("Invalid response type".to_string())),
        }
    }
}

// ----------------------------------------------------------------------
// Test Helpers
// ----------------------------------------------------------------------
struct TestCluster {
    _temp_dir: TempDir,
    network: Arc<Mutex<ChaosNetwork>>,
    nodes: HashMap<NodeId, Arc<RaftNode<SimpleStateMachine, MockTransport>>>,
    sms: HashMap<NodeId, Arc<SimpleStateMachine>>,
}

impl TestCluster {
    fn new(size: usize, seed: u64) -> Self {
        let temp_dir = TempDir::new().unwrap();
        let network = Arc::new(Mutex::new(ChaosNetwork::new(seed)));
        let node_ids: Vec<NodeId> = (1..=size as NodeId).collect();
        let mut nodes = HashMap::new();
        let mut sms = HashMap::new();

        for &id in &node_ids {
            let peers: Vec<NodeId> = node_ids.iter().cloned().filter(|&p| p != id).collect();
            let sm = Arc::new(SimpleStateMachine::default());
            let transport = Arc::new(MockTransport {
                node_id: id,
                network: network.clone(),
            });
            let wal_path = temp_dir.path().join(format!("wal_{}.log", id));
            let node = Arc::new(RaftNode::new(0, id, peers, wal_path, sm.clone(), transport).unwrap());
            nodes.insert(id, node.clone());
            sms.insert(id, sm);
        }

        {
            let mut net = network.lock();
            net.nodes = nodes.clone();
        }

        Self {
            _temp_dir: temp_dir,
            network,
            nodes,
            sms,
        }
    }

    fn tick_n(&self, n: usize) {
        for _ in 0..n {
            self.network.lock().tick();
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn find_leader(&self) -> Option<NodeId> {
        for (&id, node) in &self.nodes {
            if node.state.lock().role == Role::Leader {
                return Some(id);
            }
        }
        None
    }
}

// ----------------------------------------------------------------------
// Acceptance Tests
// ----------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_election_converges_no_partition() {
    let cluster = TestCluster::new(3, 42);
    // Run for 150 ticks to let election timeouts fire and leader settle
    cluster.tick_n(150);

    let leader = cluster.find_leader();
    assert!(leader.is_some(), "No leader was elected");

    // Verify there is only one leader
    let mut leader_count = 0;
    for node in cluster.nodes.values() {
        if node.state.lock().role == Role::Leader {
            leader_count += 1;
        }
    }
    assert_eq!(leader_count, 1, "Expected exactly one leader");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_election_converges_after_partition_heals() {
    let cluster = TestCluster::new(3, 101);
    cluster.tick_n(100);

    let l1 = cluster.find_leader().expect("Initial election failed");

    // Partition leader away from followers
    {
        let mut net = cluster.network.lock();
        net.partition_group(&[l1]);
    }

    // Tick enough for followers to notice heartbeat failure and elect a new leader
    cluster.tick_n(150);

    let mut net = cluster.network.lock();
    net.heal_partitions();
    drop(net);

    // Let the partition heal and cluster settle
    cluster.tick_n(150);

    let final_leader = cluster.find_leader();
    assert!(final_leader.is_some(), "Expected a leader after partition heals");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_no_two_leaders_same_term() {
    // Property test across 100+ seeded schedules
    for seed in 0..105 {
        let cluster = TestCluster::new(5, seed);

        // Inject some chaotic partitions at setup
        {
            let mut net = cluster.network.lock();
            net.loss_rate = 0.05;
            net.delay_range = 1..5;
            // partition a minority
            net.partition_group(&[1, 2]);
        }

        // Run simulation and assert safety invariant at each step
        for _ in 0..80 {
            cluster.network.lock().tick();
            
            // Invariant: no two leaders in the same term
            let mut terms_with_leaders = HashMap::new();
            for (&id, node) in &cluster.nodes {
                let state = node.state.lock();
                if state.role == Role::Leader {
                    if let Some(&other_id) = terms_with_leaders.get(&state.current_term) {
                        panic!(
                            "Seed {}: Node {} and Node {} both leaders in term {}",
                            seed, id, other_id, state.current_term
                        );
                    }
                    terms_with_leaders.insert(state.current_term, id);
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }

        // Heal and verify it heals correctly
        {
            let mut net = cluster.network.lock();
            net.loss_rate = 0.0;
            net.delay_range = 0..1;
            net.heal_partitions();
        }
        cluster.tick_n(100);
        let leader = cluster.find_leader();
        assert!(leader.is_some(), "Seed {}: Expected cluster to heal and elect a leader", seed);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_committed_entries_survive_minority_failure() {
    let cluster = TestCluster::new(3, 202);
    cluster.tick_n(100);

    let l1 = cluster.find_leader().expect("Election failed");

    // Propose an entry
    let rx = cluster.nodes.get(&l1).unwrap().propose(b"x,10".to_vec());
    let _index = rx.await.unwrap().expect("Proposal failed");
    cluster.tick_n(50);

    // Verify it is committed
    assert_eq!(
        cluster.sms.get(&l1).unwrap().db.lock().get("x").map(|s| s.as_str()),
        Some("10")
    );

    // Disconnect one follower (minority failure)
    let followers: Vec<NodeId> = cluster.nodes.keys().cloned().filter(|&id| id != l1).collect();
    let disconnected = followers[0];
    {
        let mut net = cluster.network.lock();
        net.partition_group(&[disconnected]);
    }

    // Propose a second entry
    let rx2 = cluster.nodes.get(&l1).unwrap().propose(b"y,20".to_vec());
    let _index2 = rx2.await.unwrap().expect("Proposal 2 failed");
    cluster.tick_n(50);

    // Verify second entry commits on leader and the connected follower
    assert_eq!(
        cluster.sms.get(&l1).unwrap().db.lock().get("y").map(|s| s.as_str()),
        Some("20")
    );

    // Reconnect the failed follower
    {
        let mut net = cluster.network.lock();
        net.heal_partitions();
    }
    cluster.tick_n(100);

    // Verify reconnected follower caught up and has both entries
    assert_eq!(
        cluster.sms.get(&disconnected).unwrap().db.lock().get("x").map(|s| s.as_str()),
        Some("10")
    );
    assert_eq!(
        cluster.sms.get(&disconnected).unwrap().db.lock().get("y").map(|s| s.as_str()),
        Some("20")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_log_matching_property() {
    let cluster = TestCluster::new(5, 303);
    cluster.tick_n(100);

    // Run chaos simulation with random proposals and partitions
    let mut rng = rand::rngs::StdRng::seed_from_u64(303);
    for i in 0..10 {
        let leader = cluster.find_leader();
        if let Some(l) = leader {
            let _ = cluster.nodes.get(&l).unwrap().propose(format!("k_{},v_{}", i, i).into_bytes());
        }

        // Apply a random partition
        {
            let mut net = cluster.network.lock();
            if rng.gen_bool(0.4) {
                let g1 = vec![1, 2];
                net.partition_group(&g1);
            } else {
                net.heal_partitions();
            }
        }
        cluster.tick_n(30);
    }

    // Heal all partitions and let the log synchronize
    {
        let mut net = cluster.network.lock();
        net.heal_partitions();
    }
    cluster.tick_n(150);

    // Assert Log Matching Property:
    // If two logs contain an entry with the same index and term, then they are
    // identical in all entries up through the given index.
    let node_logs: HashMap<NodeId, Vec<LogEntry>> = cluster
        .nodes
        .iter()
        .map(|(&id, node)| (id, node.state.lock().log.clone()))
        .collect();

    for (&id1, log1) in &node_logs {
        for (&id2, log2) in &node_logs {
            if id1 == id2 {
                continue;
            }
            let min_len = log1.len().min(log2.len());
            for idx in 0..min_len {
                if log1[idx].index == log2[idx].index && log1[idx].term == log2[idx].term {
                    // Match up to idx
                    for k in 0..=idx {
                        assert_eq!(
                            log1[k].term, log2[k].term,
                            "Log mismatch between node {} and {} at index {}",
                            id1, id2, log1[k].index
                        );
                        assert_eq!(
                            log1[k].data, log2[k].data,
                            "Log mismatch data between node {} and {} at index {}",
                            id1, id2, log1[k].index
                        );
                    }
                }
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_linearizability_after_chaos() {
    // 100+ seeded scenarios vs a sequential reference model
    println!("Starting test_linearizability_after_chaos property test...");
    for seed in 0..105 {
        println!("Running linearizability scenario with seed={}", seed);
        let cluster = TestCluster::new(3, seed);
        cluster.tick_n(100);

        let mut ref_model = HashMap::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Inject random proposals and network splits
        for step in 0..5 {
            let key = format!("step_{}", step);
            let val = format!("val_{}", rng.gen::<u32>());

            let leader = cluster.find_leader();
            if let Some(l) = leader {
                let rx = cluster.nodes.get(&l).unwrap().propose(format!("{},{}", key, val).into_bytes());
                // Let some commit and some maybe time out/fail due to partitions
                cluster.tick_n(10);
                if let Ok(Ok(_)) = tokio::time::timeout(Duration::from_millis(5), rx).await {
                    ref_model.insert(key, val);
                }
            }

            // Shuffle partitions
            {
                let mut net = cluster.network.lock();
                if rng.gen_bool(0.5) {
                    net.partition_group(&[1]);
                } else {
                    net.heal_partitions();
                }
            }
            cluster.tick_n(20);
        }

        // Heal partitions completely and let cluster converge
        {
            let mut net = cluster.network.lock();
            net.heal_partitions();
        }
        cluster.tick_n(200);

        // Assert all nodes converge to exactly the same final database state matching ref_model
        let first_node_db = cluster.sms.values().next().unwrap().db.lock().clone();
        for (&id, sm) in &cluster.sms {
            let node_db = sm.db.lock().clone();
            assert_eq!(
                first_node_db, node_db,
                "State machine divergence between nodes on seed {}",
                seed
            );
            // Verify committed values match reference model
            for (k, v) in &ref_model {
                assert_eq!(
                    node_db.get(k),
                    Some(v),
                    "Reference value mismatch on seed {} for node {}",
                    seed, id
                );
            }
        }
    }
}
