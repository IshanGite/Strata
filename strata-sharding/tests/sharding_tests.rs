use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use strata_consensus::{ConfigState, NodeId};
use strata_sharding::{
    ChaosNetwork, LoadReport, MetaCommand, MultiRaftNode, MultiRaftTransport, RangeRoute,
    RebalanceOp, RoutingTable, ShardCommand, ShardId,
};
use tempfile::TempDir;

async fn run_network_ticks(network: &Arc<Mutex<ChaosNetwork>>, ticks: usize) {
    for _ in 0..ticks {
        network.lock().tick();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

// 1. Range-based Shard Splitting Test
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_shard_split_preserves_all_keys() {
    let temp_dir = TempDir::new().unwrap();
    let network = Arc::new(Mutex::new(ChaosNetwork::new(42)));

    let node_ids = vec![1, 2, 3];
    let mut servers = HashMap::new();

    // Start 3 Multi-Raft node servers
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

    // Start Shard 1 (parent shard) on all 3 nodes
    for &id in &node_ids {
        let peers: Vec<NodeId> = node_ids.iter().cloned().filter(|&p| p != id).collect();
        servers.get(&id).unwrap().start_shard(ShardId(1), peers);
    }

    // Let the cluster elect a leader
    run_network_ticks(&network, 100).await;

    // Find leader of Shard 1
    let mut leader_id = 0;
    for &id in &node_ids {
        let shards = servers.get(&id).unwrap().shards.lock();
        if let Some(node) = shards.get(&ShardId(1)) {
            if node.state.lock().role == strata_consensus::Role::Leader {
                leader_id = id;
                break;
            }
        }
    }
    assert!(leader_id > 0, "No leader elected for Shard 1");

    // Write keys to Shard 1
    let leader_node = servers
        .get(&leader_id)
        .unwrap()
        .shards
        .lock()
        .get(&ShardId(1))
        .unwrap()
        .clone();

    let ts = strata_storage::HlcTimestamp {
        physical: 1,
        logical: 0,
    };
    let put_a = ShardCommand::Put {
        key: b"apple".to_vec(),
        value: b"red".to_vec(),
        ts,
    };
    let put_z = ShardCommand::Put {
        key: b"zebra".to_vec(),
        value: b"stripes".to_vec(),
        ts,
    };

    let rx_a = leader_node.propose(bincode::serialize(&put_a).unwrap());
    let rx_z = leader_node.propose(bincode::serialize(&put_z).unwrap());
    run_network_ticks(&network, 20).await;
    assert!(rx_a.await.unwrap().is_ok());
    assert!(rx_z.await.unwrap().is_ok());

    // Split Shard 1 at key "m" -> Shard 2
    let split_cmd = ShardCommand::Split {
        new_shard_id: ShardId(2),
        split_key: b"m".to_vec(),
    };
    let rx_split = leader_node.propose(bincode::serialize(&split_cmd).unwrap());
    run_network_ticks(&network, 20).await;
    assert!(rx_split.await.unwrap().is_ok());

    // Start Shard 2 replicas on all 3 nodes
    for &id in &node_ids {
        let peers: Vec<NodeId> = node_ids.iter().cloned().filter(|&p| p != id).collect();
        servers.get(&id).unwrap().start_shard(ShardId(2), peers);
    }

    // Run ticks to stabilize Shard 2 and elect its leader
    run_network_ticks(&network, 100).await;

    // Verify Shard 1 state machine: key "apple" is kept, "zebra" is removed
    for &id in &node_ids {
        let sms = servers.get(&id).unwrap().sms.lock();
        if let Some(sm) = sms.get(&ShardId(1)) {
            let opt = sm.storage.lock();
            let storage = opt.as_ref().unwrap();
            use strata_storage::Storage;
            let max_ts = strata_storage::HlcTimestamp {
                physical: u64::MAX,
                logical: u32::MAX,
            };
            let val_apple = storage.get(b"apple", ts).unwrap();
            let val_zebra = storage.get(b"zebra", max_ts).unwrap();
            assert_eq!(val_apple, Some(b"red".to_vec()));
            assert_eq!(val_zebra, None);
        }
    }

    // Verify Shard 2 state machine: key "zebra" was migrated successfully!
    for &id in &node_ids {
        let sms = servers.get(&id).unwrap().sms.lock();
        if let Some(sm) = sms.get(&ShardId(2)) {
            let opt = sm.storage.lock();
            let storage = opt.as_ref().unwrap();
            use strata_storage::Storage;
            let val_zebra = storage.get(b"zebra", ts).unwrap();
            let val_apple = storage.get(b"apple", ts).unwrap();
            assert_eq!(val_zebra, Some(b"stripes".to_vec()));
            assert_eq!(val_apple, None);
        }
    }
}

// 2. Range-based Shard Merging Test
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_shard_merge_preserves_all_keys() {
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

    // Start Shard 1 and Shard 2
    for &id in &node_ids {
        let peers: Vec<NodeId> = node_ids.iter().cloned().filter(|&p| p != id).collect();
        servers
            .get(&id)
            .unwrap()
            .start_shard(ShardId(1), peers.clone());
        servers.get(&id).unwrap().start_shard(ShardId(2), peers);
    }

    run_network_ticks(&network, 100).await;

    // Find leaders
    let mut leader1 = 0;
    let mut leader2 = 0;
    for &id in &node_ids {
        let shards = servers.get(&id).unwrap().shards.lock();
        if let Some(node) = shards.get(&ShardId(1)) {
            if node.state.lock().role == strata_consensus::Role::Leader {
                leader1 = id;
            }
        }
        if let Some(node) = shards.get(&ShardId(2)) {
            if node.state.lock().role == strata_consensus::Role::Leader {
                leader2 = id;
            }
        }
    }
    assert!(leader1 > 0 && leader2 > 0);

    let ts = strata_storage::HlcTimestamp {
        physical: 1,
        logical: 0,
    };
    let put_a = ShardCommand::Put {
        key: b"apple".to_vec(),
        value: b"red".to_vec(),
        ts,
    };
    let put_z = ShardCommand::Put {
        key: b"zebra".to_vec(),
        value: b"stripes".to_vec(),
        ts,
    };

    // Write "apple" to Shard 1, "zebra" to Shard 2
    let node1 = servers
        .get(&leader1)
        .unwrap()
        .shards
        .lock()
        .get(&ShardId(1))
        .unwrap()
        .clone();
    let node2 = servers
        .get(&leader2)
        .unwrap()
        .shards
        .lock()
        .get(&ShardId(2))
        .unwrap()
        .clone();

    let _ = node1
        .propose(bincode::serialize(&put_a).unwrap())
        .await
        .unwrap();
    let _ = node2
        .propose(bincode::serialize(&put_z).unwrap())
        .await
        .unwrap();
    run_network_ticks(&network, 20).await;

    // Merge Shard 2 into Shard 1
    let merge_cmd = ShardCommand::Merge {
        target_shard_id: ShardId(1),
    };
    let _ = node2
        .propose(bincode::serialize(&merge_cmd).unwrap())
        .await
        .unwrap();
    run_network_ticks(&network, 20).await;

    // Now start/simulate the merge collection on Shard 1
    for &id in &node_ids {
        let server = servers.get(&id).unwrap();
        // Stop Shard 2
        server.stop_shard(ShardId(2));

        // Re-open Shard 1 database to trigger merge file loading
        server.stop_shard(ShardId(1));
        let peers: Vec<NodeId> = node_ids.iter().cloned().filter(|&p| p != id).collect();
        server.start_shard(ShardId(1), peers);
    }

    run_network_ticks(&network, 100).await;

    // Verify Shard 1 now contains both "apple" and "zebra"
    for &id in &node_ids {
        let sms = servers.get(&id).unwrap().sms.lock();
        if let Some(sm) = sms.get(&ShardId(1)) {
            let opt = sm.storage.lock();
            let storage = opt.as_ref().unwrap();
            use strata_storage::Storage;
            let val_apple = storage.get(b"apple", ts).unwrap();
            let val_zebra = storage.get(b"zebra", ts).unwrap();
            assert_eq!(val_apple, Some(b"red".to_vec()));
            assert_eq!(val_zebra, Some(b"stripes".to_vec()));
        }
    }
}

// 3. Joint Consensus Membership Change Test
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_joint_consensus_membership_change_no_split_brain() {
    let temp_dir = TempDir::new().unwrap();
    let network = Arc::new(Mutex::new(ChaosNetwork::new(42)));

    // Cluster starts with nodes [1, 2, 3]
    let initial_nodes = vec![1, 2, 3];
    let all_nodes = vec![1, 2, 3, 4];
    let mut servers = HashMap::new();

    for &id in &all_nodes {
        let node_dir = temp_dir.path().join(format!("node_{}", id));
        let transport = Arc::new(MultiRaftTransport {
            node_id: id,
            network: network.clone(),
        });
        let server = Arc::new(MultiRaftNode::new(id, node_dir, transport));
        servers.insert(id, server.clone());
        network.lock().node_servers.insert(id, server);
    }

    // Start Shard 1 on nodes [1, 2, 3]
    for &id in &initial_nodes {
        let peers: Vec<NodeId> = initial_nodes.iter().cloned().filter(|&p| p != id).collect();
        let server = servers.get(&id).unwrap();
        server.start_shard(ShardId(1), peers);
    }

    run_network_ticks(&network, 100).await;

    // Propose membership change to include node 4: new_nodes = [2, 3, 4]
    let mut leader_id = 0;
    for &id in &initial_nodes {
        let shards = servers.get(&id).unwrap().shards.lock();
        if let Some(node) = shards.get(&ShardId(1)) {
            if node.state.lock().role == strata_consensus::Role::Leader {
                leader_id = id;
                break;
            }
        }
    }
    assert!(leader_id > 0);

    // Start Shard 1 replica on node 4 so it's ready to receive replication
    let peers4: Vec<NodeId> = initial_nodes.iter().cloned().collect();
    servers.get(&4).unwrap().start_shard(ShardId(1), peers4);

    let leader_node = servers
        .get(&leader_id)
        .unwrap()
        .shards
        .lock()
        .get(&ShardId(1))
        .unwrap()
        .clone();
    let rx_change = leader_node.change_membership(vec![2, 3, 4]);

    // Let joint consensus phase begin and progress
    run_network_ticks(&network, 100).await;

    // Verify change completes successfully
    assert!(rx_change.await.unwrap().is_ok());

    // Check that config is stable with [2, 3, 4]
    for &id in &[2, 3, 4] {
        let shards = servers.get(&id).unwrap().shards.lock();
        if let Some(node) = shards.get(&ShardId(1)) {
            let state = node.state.lock();
            assert_eq!(state.config, ConfigState::Stable(vec![2, 3, 4]));
        }
    }
}

// 4. Greedy Rebalancer Convergence Test
#[tokio::test]
async fn test_rebalancer_converges() {
    let mut table = RoutingTable::new();
    table.routes = vec![
        RangeRoute {
            start_key: b"a".to_vec(),
            end_key: b"m".to_vec(),
            shard_id: ShardId(1),
            raft_group: vec![1],
        },
        RangeRoute {
            start_key: b"m".to_vec(),
            end_key: b"s".to_vec(),
            shard_id: ShardId(2),
            raft_group: vec![1],
        },
        RangeRoute {
            start_key: b"s".to_vec(),
            end_key: vec![],
            shard_id: ShardId(3),
            raft_group: vec![2],
        },
    ];

    let current_load = LoadReport {
        shard_loads: vec![(ShardId(1), 10), (ShardId(2), 10), (ShardId(3), 10)],
    };

    let all_nodes = vec![1, 2, 3];
    let ops = table.propose_rebalance(&current_load, &all_nodes, 5);
    assert!(
        !ops.is_empty(),
        "Imbalance should trigger rebalance operations"
    );

    if let RebalanceOp::Move { shard, from, to } = &ops[0] {
        assert!(shard.0 == 1 || shard.0 == 2);
        assert_eq!(*from, 1);
        assert_eq!(*to, 3);
    } else {
        panic!("First op should be a Move");
    }
}

// 5. Routing Table Consistency Test
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_routing_table_consistency() {
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

    // Start routing table meta-Raft group (Shard 0) on all nodes
    for &id in &node_ids {
        let peers: Vec<NodeId> = node_ids.iter().cloned().filter(|&p| p != id).collect();
        let server = servers.get(&id).unwrap();
        server.start_shard(ShardId(0), peers);
    }

    run_network_ticks(&network, 100).await;

    // Propose range updates to meta-Raft group
    let mut leader_id = 0;
    for &id in &node_ids {
        let shards = servers.get(&id).unwrap().shards.lock();
        if let Some(node) = shards.get(&ShardId(0)) {
            if node.state.lock().role == strata_consensus::Role::Leader {
                leader_id = id;
                break;
            }
        }
    }
    assert!(leader_id > 0);

    let leader_node = servers
        .get(&leader_id)
        .unwrap()
        .shards
        .lock()
        .get(&ShardId(0))
        .unwrap()
        .clone();

    let route = RangeRoute {
        start_key: b"a".to_vec(),
        end_key: b"z".to_vec(),
        shard_id: ShardId(1),
        raft_group: vec![1, 2, 3],
    };
    let update_cmd = MetaCommand::UpdateRoute(route);
    let rx_route = leader_node.propose(bincode::serialize(&update_cmd).unwrap());

    run_network_ticks(&network, 20).await;
    assert!(rx_route.await.unwrap().is_ok());

    // Verify routing table's view matches proposed metadata across nodes
    for &id in &node_ids {
        let server = servers.get(&id).unwrap();
        let final_table = server.table.lock();
        assert_eq!(final_table.routes.len(), 1);
        assert_eq!(final_table.routes[0].shard_id.0, 1);
        assert_eq!(final_table.routes[0].raft_group, vec![1, 2, 3]);
    }
}
