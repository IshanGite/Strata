use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use strata_consensus::NodeId;
use strata_sharding::{
    ChaosNetwork, MultiRaftNode, MultiRaftTransport, RangeRoute, RoutingTable, ShardId, lock_key,
};
use strata_txn::{DistributedTxnCoordinator, Hlc, HlcTimestamp, Mutation, TransactionCoordinator, TxnError};
use strata_storage::Storage;
use tempfile::TempDir;

// 1. test_hlc_causality — events with a happens-before relationship get
//    HLC timestamps respecting that order across simulated network delay.
#[test]
fn test_hlc_causality() {
    let clock1_time = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(10));
    let c1 = clock1_time.clone();
    let hlc1 = Hlc::new_with_clock(0, 0, Box::new(move || c1.load(std::sync::atomic::Ordering::SeqCst)));

    let clock2_time = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(8)); // Clock drift backward!
    let c2 = clock2_time.clone();
    let hlc2 = Hlc::new_with_clock(0, 0, Box::new(move || c2.load(std::sync::atomic::Ordering::SeqCst)));

    // Event 1 on node 1
    let ts1 = hlc1.local_event();
    assert_eq!(ts1.physical, 10);
    assert_eq!(ts1.logical, 0);

    // Event 2 on node 1 (happens-after Event 1)
    let ts2 = hlc1.local_event();
    assert_eq!(ts2.physical, 10);
    assert_eq!(ts2.logical, 1);
    assert!(ts2 > ts1);

    // Message sent from node 1 (at ts2) to node 2
    // Node 2 receives message. Node 2's local physical time is 8, message physical is 10.
    let ts3 = hlc2.receive(ts2);
    // Node 2 physical should catch up to max(8, 10) = 10, and increment logical
    assert_eq!(ts3.physical, 10);
    assert_eq!(ts3.logical, 2);
    assert!(ts3 > ts2);

    // Node 2 has local event, physical clock now advanced to 15
    clock2_time.store(15, std::sync::atomic::Ordering::SeqCst);
    let ts4 = hlc2.local_event();
    assert_eq!(ts4.physical, 15);
    assert_eq!(ts4.logical, 0);
    assert!(ts4 > ts3);
}

// 2. test_cross_shard_txn_atomic_commit — a transaction writing to 3
//    different shards either commits fully on all 3 or is visible on none.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_cross_shard_txn_atomic_commit() {
    let temp_dir = TempDir::new().unwrap();
    let network = Arc::new(Mutex::new(ChaosNetwork::new(42)));
    let node_ids = vec![1, 2, 3];
    let mut servers = HashMap::new();

    for &id in &node_ids {
        let node_dir = temp_dir.path().join(format!("node_{}", id));
        let transport = Arc::new(MultiRaftTransport {
            node_id: id,
            network: network.clone(),
        });
        let server = Arc::new(MultiRaftNode::new(id, node_dir, transport));
        servers.insert(id, server.clone());
        network.lock().node_servers.insert(id, server);
    }

    let mut table = RoutingTable::new();
    table.routes = vec![
        RangeRoute {
            start_key: b"a".to_vec(),
            end_key: b"k".to_vec(),
            shard_id: ShardId(1),
            raft_group: vec![1, 2, 3],
        },
        RangeRoute {
            start_key: b"k".to_vec(),
            end_key: b"s".to_vec(),
            shard_id: ShardId(2),
            raft_group: vec![1, 2, 3],
        },
        RangeRoute {
            start_key: b"s".to_vec(),
            end_key: b"z".to_vec(),
            shard_id: ShardId(3),
            raft_group: vec![1, 2, 3],
        },
    ];

    for &id in &node_ids {
        let server = servers.get(&id).unwrap();
        *server.table.lock() = table.clone();
        
        let peers: Vec<NodeId> = vec![1, 2, 3].into_iter().filter(|&p| p != id).collect();
        server.start_shard(ShardId(1), peers.clone());
        server.start_shard(ShardId(2), peers.clone());
        server.start_shard(ShardId(3), peers);
    }

    // Run ticker in background
    let network_clone = network.clone();
    let ticker = tokio::spawn(async move {
        loop {
            network_clone.lock().tick();
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    });

    // Let the cluster elect leaders
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let hlc = Arc::new(Hlc::new(100, 0));
    let node_servers = Arc::new(parking_lot::Mutex::new(servers));
    let coordinator = DistributedTxnCoordinator::new(
        hlc.clone(),
        node_servers.clone(),
        Arc::new(parking_lot::Mutex::new(table.clone())),
    );

    // Start Transaction 1
    let start_ts = coordinator.begin();
    let mutations = vec![
        Mutation::Put(b"apple".to_vec(), b"red".to_vec()),
        Mutation::Put(b"lemon".to_vec(), b"yellow".to_vec()),
        Mutation::Put(b"zebra".to_vec(), b"stripes".to_vec()),
    ];

    coordinator.prewrite(start_ts, &mutations).unwrap();

    let commit_ts = hlc.local_event();
    coordinator.commit(start_ts, commit_ts).unwrap();

    // Verify reads
    let val_apple = coordinator.get(b"apple", commit_ts).await.unwrap();
    let val_lemon = coordinator.get(b"lemon", commit_ts).await.unwrap();
    let val_zebra = coordinator.get(b"zebra", commit_ts).await.unwrap();

    assert_eq!(val_apple, Some(b"red".to_vec()));
    assert_eq!(val_lemon, Some(b"yellow".to_vec()));
    assert_eq!(val_zebra, Some(b"stripes".to_vec()));

    // Transaction 2: write new values but abort
    let start_ts2 = coordinator.begin();
    let mutations2 = vec![
        Mutation::Put(b"apple".to_vec(), b"green".to_vec()),
        Mutation::Put(b"lemon".to_vec(), b"sour".to_vec()),
    ];

    coordinator.prewrite(start_ts2, &mutations2).unwrap();
    coordinator.abort(start_ts2).unwrap();

    // Verify still old values are read
    let val_apple2 = coordinator.get(b"apple", hlc.local_event()).await.unwrap();
    let val_lemon2 = coordinator.get(b"lemon", hlc.local_event()).await.unwrap();
    assert_eq!(val_apple2, Some(b"red".to_vec()));
    assert_eq!(val_lemon2, Some(b"yellow".to_vec()));

    ticker.abort();
}

// 3. test_write_write_conflict_detected — concurrent transactions on the
//    same key, one aborts, verified via prewrite failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_write_write_conflict_detected() {
    let temp_dir = TempDir::new().unwrap();
    let network = Arc::new(Mutex::new(ChaosNetwork::new(42)));
    let node_ids = vec![1, 2, 3];
    let mut servers = HashMap::new();

    for &id in &node_ids {
        let node_dir = temp_dir.path().join(format!("node_{}", id));
        let transport = Arc::new(MultiRaftTransport {
            node_id: id,
            network: network.clone(),
        });
        let server = Arc::new(MultiRaftNode::new(id, node_dir, transport));
        servers.insert(id, server.clone());
        network.lock().node_servers.insert(id, server);
    }

    let mut table = RoutingTable::new();
    table.routes = vec![
        RangeRoute {
            start_key: b"a".to_vec(),
            end_key: b"z".to_vec(),
            shard_id: ShardId(1),
            raft_group: vec![1, 2, 3],
        },
    ];

    for &id in &node_ids {
        let server = servers.get(&id).unwrap();
        *server.table.lock() = table.clone();
        
        let peers: Vec<NodeId> = vec![1, 2, 3].into_iter().filter(|&p| p != id).collect();
        server.start_shard(ShardId(1), peers);
    }

    let network_clone = network.clone();
    let ticker = tokio::spawn(async move {
        loop {
            network_clone.lock().tick();
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    });

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let hlc = Arc::new(Hlc::new(100, 0));
    let node_servers = Arc::new(parking_lot::Mutex::new(servers));
    let coordinator = DistributedTxnCoordinator::new(
        hlc.clone(),
        node_servers.clone(),
        Arc::new(parking_lot::Mutex::new(table.clone())),
    );

    // Scenario A: Lock Conflict
    let start_ts1 = coordinator.begin();
    let start_ts2 = coordinator.begin(); // concurrent transaction

    // T1 prewrites key
    let muts1 = vec![Mutation::Put(b"apple".to_vec(), b"red".to_vec())];
    coordinator.prewrite(start_ts1, &muts1).unwrap();

    // T2 tries to prewrite same key -> should fail with LockConflict
    let muts2 = vec![Mutation::Put(b"apple".to_vec(), b"green".to_vec())];
    let res = coordinator.prewrite(start_ts2, &muts2);
    assert!(res.is_err());
    assert!(matches!(res.unwrap_err(), TxnError::LockConflict { .. }));

    // Abort T1 to release lock
    coordinator.abort(start_ts1).unwrap();

    // Scenario B: Write Conflict
    // Commit a write at commit_ts
    let start_ts3 = coordinator.begin();
    let muts3 = vec![Mutation::Put(b"apple".to_vec(), b"yellow".to_vec())];
    coordinator.prewrite(start_ts3, &muts3).unwrap();
    let commit_ts3 = hlc.local_event();
    coordinator.commit(start_ts3, commit_ts3).unwrap();

    // T2 tries to prewrite -> should fail with WriteConflict because a newer version exists committed at commit_ts3
    let res2 = coordinator.prewrite(start_ts2, &muts2);
    assert!(res2.is_err());
    assert!(matches!(res2.unwrap_err(), TxnError::WriteConflict { .. }));

    ticker.abort();
}

// 4. test_read_encounters_lock_resolves_correctly — a read racing an
//    in-flight transaction's lock either waits or correctly determines
//    committed/aborted state, never returns a phantom partial write.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_read_encounters_lock_resolves_correctly() {
    let temp_dir = TempDir::new().unwrap();
    let network = Arc::new(Mutex::new(ChaosNetwork::new(42)));
    let node_ids = vec![1, 2, 3];
    let mut servers = HashMap::new();

    for &id in &node_ids {
        let node_dir = temp_dir.path().join(format!("node_{}", id));
        let transport = Arc::new(MultiRaftTransport {
            node_id: id,
            network: network.clone(),
        });
        let server = Arc::new(MultiRaftNode::new(id, node_dir, transport));
        servers.insert(id, server.clone());
        network.lock().node_servers.insert(id, server);
    }

    let mut table = RoutingTable::new();
    table.routes = vec![
        RangeRoute {
            start_key: b"a".to_vec(),
            end_key: b"z".to_vec(),
            shard_id: ShardId(1),
            raft_group: vec![1, 2, 3],
        },
    ];

    for &id in &node_ids {
        let server = servers.get(&id).unwrap();
        *server.table.lock() = table.clone();
        
        let peers: Vec<NodeId> = vec![1, 2, 3].into_iter().filter(|&p| p != id).collect();
        server.start_shard(ShardId(1), peers);
    }

    let network_clone = network.clone();
    let ticker = tokio::spawn(async move {
        loop {
            network_clone.lock().tick();
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    });

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let hlc = Arc::new(Hlc::new(100, 0));
    let node_servers = Arc::new(parking_lot::Mutex::new(servers));
    let coordinator = Arc::new(DistributedTxnCoordinator::new(
        hlc.clone(),
        node_servers.clone(),
        Arc::new(parking_lot::Mutex::new(table.clone())),
    ));

    // 1. Prewrite T1
    let start_ts1 = coordinator.begin();
    let muts1 = vec![Mutation::Put(b"apple".to_vec(), b"red".to_vec())];
    coordinator.prewrite(start_ts1, &muts1).unwrap();

    // Spawn a reader reading as of a timestamp after start_ts1
    let reader_ts = hlc.local_event();
    let coordinator_clone = coordinator.clone();
    let read_handle = tokio::spawn(async move {
        coordinator_clone.get(b"apple", reader_ts).await
    });

    // Wait a bit, verify reader is still waiting
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert!(!read_handle.is_finished());

    // Commit T1
    let commit_ts1 = hlc.local_event();
    coordinator.commit(start_ts1, commit_ts1).unwrap();

    // Now reader should finish and see None (since reader_ts < commit_ts1)
    let read_res = read_handle.await.unwrap().unwrap();
    assert_eq!(read_res, None);

    // Let's do another read as of after commit_ts1
    let reader_ts2 = hlc.local_event();
    let read_res2 = coordinator.get(b"apple", reader_ts2).await.unwrap();
    assert_eq!(read_res2, Some(b"red".to_vec()));

    ticker.abort();
}

// 5. test_txn_survives_coordinator_crash_mid_commit — kill the process
//    issuing commit() after primary commits but before secondaries resolve;
//    verify a later reader can still resolve the transaction to its correct
//    final state by consulting the primary.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_txn_survives_coordinator_crash_mid_commit() {
    let temp_dir = TempDir::new().unwrap();
    let network = Arc::new(Mutex::new(ChaosNetwork::new(42)));
    let node_ids = vec![1, 2, 3];
    let mut servers = HashMap::new();

    for &id in &node_ids {
        let node_dir = temp_dir.path().join(format!("node_{}", id));
        let transport = Arc::new(MultiRaftTransport {
            node_id: id,
            network: network.clone(),
        });
        let server = Arc::new(MultiRaftNode::new(id, node_dir, transport));
        servers.insert(id, server.clone());
        network.lock().node_servers.insert(id, server);
    }

    let mut table = RoutingTable::new();
    table.routes = vec![
        RangeRoute {
            start_key: b"a".to_vec(),
            end_key: b"k".to_vec(),
            shard_id: ShardId(1),
            raft_group: vec![1, 2, 3],
        },
        RangeRoute {
            start_key: b"k".to_vec(),
            end_key: b"z".to_vec(),
            shard_id: ShardId(2),
            raft_group: vec![1, 2, 3],
        },
    ];

    for &id in &node_ids {
        let server = servers.get(&id).unwrap();
        *server.table.lock() = table.clone();
        
        let peers: Vec<NodeId> = vec![1, 2, 3].into_iter().filter(|&p| p != id).collect();
        server.start_shard(ShardId(1), peers.clone());
        server.start_shard(ShardId(2), peers);
    }

    let network_clone = network.clone();
    let ticker = tokio::spawn(async move {
        loop {
            network_clone.lock().tick();
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    });

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Use a manual clock
    let manual_time = Arc::new(std::sync::atomic::AtomicU64::new(100));
    let m_time = manual_time.clone();
    let hlc = Arc::new(Hlc::new_with_clock(
        100,
        0,
        Box::new(move || m_time.load(std::sync::atomic::Ordering::SeqCst)),
    ));
    
    let node_servers = Arc::new(parking_lot::Mutex::new(servers));
    let coordinator = Arc::new(DistributedTxnCoordinator::new(
        hlc.clone(),
        node_servers.clone(),
        Arc::new(parking_lot::Mutex::new(table.clone())),
    ));

    // Start Transaction: primary "apple" (Shard 1), secondary "lemon" (Shard 2)
    let start_ts = coordinator.begin();
    let mutations = vec![
        Mutation::Put(b"apple".to_vec(), b"red".to_vec()),
        Mutation::Put(b"lemon".to_vec(), b"yellow".to_vec()),
    ];

    // Prewrite both keys
    coordinator.prewrite(start_ts, &mutations).unwrap();

    // Commit primary key first
    let commit_ts = hlc.local_event();
    let primary_cmd = strata_sharding::ShardCommand::TxnCommit {
        key: b"apple".to_vec(),
        start_ts,
        commit_ts,
        is_primary: true,
    };
    coordinator.propose_command(b"apple", primary_cmd).await.unwrap();

    // Simulate coordinator crash: we do NOT commit the secondary key "lemon"!
    // Advance manual time so that the lock becomes stale
    manual_time.store(2000, std::sync::atomic::Ordering::SeqCst);

    // Now, perform a read on "lemon" as of commit_ts or after.
    // The reader will find the lock on "lemon". Since current time (2000) > lock.ts (100) + ttl (1000),
    // the reader will see it's stale, check the primary "apple", see that "apple" committed,
    // and roll the lock on "lemon" forward.
    let read_ts = hlc.local_event();
    let read_val = coordinator.get(b"lemon", read_ts).await.unwrap();

    // Reader should successfully resolve and return the committed value "yellow"!
    assert_eq!(read_val, Some(b"yellow".to_vec()));

    // Also verify that the secondary lock is physically removed now
    let lk = lock_key(b"lemon");
    let max_ts = HlcTimestamp { physical: u64::MAX, logical: u32::MAX };
    let lock_check = coordinator.get_storage(b"lemon").unwrap().lock().as_ref().unwrap().get(&lk, max_ts).unwrap();
    assert!(lock_check.is_none());

    ticker.abort();
}
