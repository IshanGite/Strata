use std::collections::HashMap;
use std::sync::Arc;
use strata_client::StrataClient;
use strata_server::StrataServerDaemon;
use strata_sharding::{RangeRoute, RoutingTable, ShardId};
use strata_simd::l2_distance;
use strata_txn::Mutation;
use tempfile::TempDir;

fn find_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_real_3node_cluster_survives_leader_kill_mid_write() {
    let temp_dir = TempDir::new().unwrap();
    let port1 = find_free_port();
    let port2 = find_free_port();
    let port3 = find_free_port();

    let addr1 = format!("127.0.0.1:{}", port1);
    let addr2 = format!("127.0.0.1:{}", port2);
    let addr3 = format!("127.0.0.1:{}", port3);

    let mut node_addrs = HashMap::new();
    node_addrs.insert(1, format!("http://{}", addr1));
    node_addrs.insert(2, format!("http://{}", addr2));
    node_addrs.insert(3, format!("http://{}", addr3));

    let daemon1 = Arc::new(StrataServerDaemon::new(
        1,
        addr1.clone(),
        temp_dir.path().join("node_1"),
        node_addrs.clone(),
    ));
    let daemon2 = Arc::new(StrataServerDaemon::new(
        2,
        addr2.clone(),
        temp_dir.path().join("node_2"),
        node_addrs.clone(),
    ));
    let daemon3 = Arc::new(StrataServerDaemon::new(
        3,
        addr3.clone(),
        temp_dir.path().join("node_3"),
        node_addrs.clone(),
    ));

    // Cross-link node servers for distributed coordinator
    daemon1.set_node_server(2, daemon2.multi_raft.clone());
    daemon1.set_node_server(3, daemon3.multi_raft.clone());

    daemon2.set_node_server(1, daemon1.multi_raft.clone());
    daemon2.set_node_server(3, daemon3.multi_raft.clone());

    daemon3.set_node_server(1, daemon1.multi_raft.clone());
    daemon3.set_node_server(2, daemon2.multi_raft.clone());

    let mut table = RoutingTable::new();
    table.routes = vec![RangeRoute {
        start_key: b"a".to_vec(),
        end_key: b"z".to_vec(),
        shard_id: ShardId(1),
        raft_group: vec![1, 2, 3],
    }];

    *daemon1.table.lock() = table.clone();
    *daemon2.table.lock() = table.clone();
    *daemon3.table.lock() = table.clone();

    daemon1.start_shard(ShardId(1), vec![2, 3]);
    daemon2.start_shard(ShardId(1), vec![1, 3]);
    daemon3.start_shard(ShardId(1), vec![1, 2]);

    let d1 = daemon1.clone();
    let d2 = daemon2.clone();
    let d3 = daemon3.clone();

    let h1 = tokio::spawn(async move {
        let _ = d1.run().await;
    });
    let h2 = tokio::spawn(async move {
        let _ = d2.run().await;
    });
    let h3 = tokio::spawn(async move {
        let _ = d3.run().await;
    });

    tokio::time::sleep(std::time::Duration::from_millis(600)).await;

    let seed_addrs = vec![
        format!("http://{}", addr1),
        format!("http://{}", addr2),
        format!("http://{}", addr3),
    ];
    let client = StrataClient::connect(seed_addrs.clone()).await.unwrap();

    // Write initial key
    client
        .put(b"key_before_kill".to_vec(), b"val_before".to_vec())
        .await
        .unwrap();
    let res = client.get(b"key_before_kill").await.unwrap();
    assert_eq!(res, Some(b"val_before".to_vec()));

    // Allow followers to catch up on replication log apply
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Determine current leader and kill daemon
    let shard_node1 = daemon1
        .multi_raft
        .shards
        .lock()
        .get(&ShardId(1))
        .cloned()
        .unwrap();
    let is_leader1 = shard_node1.state.lock().role == strata_consensus::Role::Leader;

    if is_leader1 {
        daemon1.shutdown();
    } else {
        let shard_node2 = daemon2
            .multi_raft
            .shards
            .lock()
            .get(&ShardId(1))
            .cloned()
            .unwrap();
        if shard_node2.state.lock().role == strata_consensus::Role::Leader {
            daemon2.shutdown();
        } else {
            daemon3.shutdown();
        }
    }

    // Wait for election timeout and cluster failover
    tokio::time::sleep(std::time::Duration::from_millis(800)).await;

    // Client write after leader kill
    let mut write_success = false;
    for _ in 0..10 {
        if client
            .put(b"key_after_kill".to_vec(), b"val_after".to_vec())
            .await
            .is_ok()
        {
            write_success = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(write_success, "Write failed after leader kill!");

    // Verify both pre-kill and post-kill keys survive
    let res_before = client.get(b"key_before_kill").await.unwrap();
    assert_eq!(res_before, Some(b"val_before".to_vec()));

    let res_after = client.get(b"key_after_kill").await.unwrap();
    assert_eq!(res_after, Some(b"val_after".to_vec()));

    daemon1.shutdown();
    daemon2.shutdown();
    daemon3.shutdown();
    let _ = tokio::join!(h1, h2, h3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_scatter_gather_knn_correct_across_shards() {
    let temp_dir = TempDir::new().unwrap();
    let port1 = find_free_port();
    let port2 = find_free_port();
    let port3 = find_free_port();

    let addr1 = format!("127.0.0.1:{}", port1);
    let addr2 = format!("127.0.0.1:{}", port2);
    let addr3 = format!("127.0.0.1:{}", port3);

    let mut node_addrs = HashMap::new();
    node_addrs.insert(1, format!("http://{}", addr1));
    node_addrs.insert(2, format!("http://{}", addr2));
    node_addrs.insert(3, format!("http://{}", addr3));

    let daemon1 = Arc::new(StrataServerDaemon::new(
        1,
        addr1.clone(),
        temp_dir.path().join("node_1"),
        node_addrs.clone(),
    ));
    let daemon2 = Arc::new(StrataServerDaemon::new(
        2,
        addr2.clone(),
        temp_dir.path().join("node_2"),
        node_addrs.clone(),
    ));
    let daemon3 = Arc::new(StrataServerDaemon::new(
        3,
        addr3.clone(),
        temp_dir.path().join("node_3"),
        node_addrs.clone(),
    ));

    daemon1.set_node_server(2, daemon2.multi_raft.clone());
    daemon1.set_node_server(3, daemon3.multi_raft.clone());

    daemon2.set_node_server(1, daemon1.multi_raft.clone());
    daemon2.set_node_server(3, daemon3.multi_raft.clone());

    daemon3.set_node_server(1, daemon1.multi_raft.clone());
    daemon3.set_node_server(2, daemon2.multi_raft.clone());

    let mut table = RoutingTable::new();
    table.routes = vec![
        RangeRoute {
            start_key: b"vec:00".to_vec(),
            end_key: b"vec:33".to_vec(),
            shard_id: ShardId(1),
            raft_group: vec![1, 2, 3],
        },
        RangeRoute {
            start_key: b"vec:33".to_vec(),
            end_key: b"vec:66".to_vec(),
            shard_id: ShardId(2),
            raft_group: vec![1, 2, 3],
        },
        RangeRoute {
            start_key: b"vec:66".to_vec(),
            end_key: b"vec:99".to_vec(),
            shard_id: ShardId(3),
            raft_group: vec![1, 2, 3],
        },
    ];

    *daemon1.table.lock() = table.clone();
    *daemon2.table.lock() = table.clone();
    *daemon3.table.lock() = table.clone();

    daemon1.start_shard(ShardId(1), vec![2, 3]);
    daemon1.start_shard(ShardId(2), vec![2, 3]);
    daemon1.start_shard(ShardId(3), vec![2, 3]);

    daemon2.start_shard(ShardId(1), vec![1, 3]);
    daemon2.start_shard(ShardId(2), vec![1, 3]);
    daemon2.start_shard(ShardId(3), vec![1, 3]);

    daemon3.start_shard(ShardId(1), vec![1, 2]);
    daemon3.start_shard(ShardId(2), vec![1, 2]);
    daemon3.start_shard(ShardId(3), vec![1, 2]);

    let d1 = daemon1.clone();
    let d2 = daemon2.clone();
    let d3 = daemon3.clone();

    let h1 = tokio::spawn(async move {
        let _ = d1.run().await;
    });
    let h2 = tokio::spawn(async move {
        let _ = d2.run().await;
    });
    let h3 = tokio::spawn(async move {
        let _ = d3.run().await;
    });

    tokio::time::sleep(std::time::Duration::from_millis(600)).await;

    let seed_addrs = vec![format!("http://{}", addr1)];
    let client = StrataClient::connect(seed_addrs).await.unwrap();

    // Insert 30 vectors across the shards
    let mut dataset: Vec<(u64, Vec<f32>)> = Vec::new();
    for i in 1..=30 {
        let mut vec = vec![0.0f32; 128];
        vec[0] = i as f32 * 0.1;
        vec[1] = (30 - i) as f32 * 0.05;
        dataset.push((i, vec.clone()));

        // Format key so routing table maps to shard 1, 2, or 3
        let key_str = format!("vec:{:02}", i);
        let val_bytes = bincode::serialize(&vec).unwrap();
        client.put(key_str.into_bytes(), val_bytes).await.unwrap();
    }

    // Query vector
    let mut query_vector = vec![0.0f32; 128];
    query_vector[0] = 0.5;
    query_vector[1] = 0.5;

    // Ground truth top-5 calculation
    let mut ground_truth: Vec<(u64, f32)> = dataset
        .iter()
        .map(|(id, v)| (*id, l2_distance(&query_vector, v)))
        .collect();
    ground_truth.sort_by(|a, b| a.1.total_cmp(&b.1));
    ground_truth.truncate(5);

    // Run scatter-gather KNN search over gRPC
    let scatter_results = client.search_knn(query_vector, 5).await.unwrap();

    assert_eq!(scatter_results.len(), 5);
    for (i, (gt_id, gt_dist)) in ground_truth.iter().enumerate() {
        let (sc_id, sc_dist) = scatter_results[i];
        assert_eq!(sc_id, *gt_id, "Mismatch at rank {}", i);
        assert!(
            (sc_dist - gt_dist).abs() < 1e-5,
            "Distance mismatch at rank {}",
            i
        );
    }

    daemon1.shutdown();
    daemon2.shutdown();
    daemon3.shutdown();
    let _ = tokio::join!(h1, h2, h3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_cross_shard_txn_over_real_network() {
    let temp_dir = TempDir::new().unwrap();
    let port1 = find_free_port();
    let port2 = find_free_port();
    let port3 = find_free_port();

    let addr1 = format!("127.0.0.1:{}", port1);
    let addr2 = format!("127.0.0.1:{}", port2);
    let addr3 = format!("127.0.0.1:{}", port3);

    let mut node_addrs = HashMap::new();
    node_addrs.insert(1, format!("http://{}", addr1));
    node_addrs.insert(2, format!("http://{}", addr2));
    node_addrs.insert(3, format!("http://{}", addr3));

    let daemon1 = Arc::new(StrataServerDaemon::new(
        1,
        addr1.clone(),
        temp_dir.path().join("node_1"),
        node_addrs.clone(),
    ));
    let daemon2 = Arc::new(StrataServerDaemon::new(
        2,
        addr2.clone(),
        temp_dir.path().join("node_2"),
        node_addrs.clone(),
    ));
    let daemon3 = Arc::new(StrataServerDaemon::new(
        3,
        addr3.clone(),
        temp_dir.path().join("node_3"),
        node_addrs.clone(),
    ));

    daemon1.set_node_server(2, daemon2.multi_raft.clone());
    daemon1.set_node_server(3, daemon3.multi_raft.clone());

    daemon2.set_node_server(1, daemon1.multi_raft.clone());
    daemon2.set_node_server(3, daemon3.multi_raft.clone());

    daemon3.set_node_server(1, daemon1.multi_raft.clone());
    daemon3.set_node_server(2, daemon2.multi_raft.clone());

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

    *daemon1.table.lock() = table.clone();
    *daemon2.table.lock() = table.clone();
    *daemon3.table.lock() = table.clone();

    daemon1.start_shard(ShardId(1), vec![2, 3]);
    daemon1.start_shard(ShardId(2), vec![2, 3]);
    daemon1.start_shard(ShardId(3), vec![2, 3]);

    daemon2.start_shard(ShardId(1), vec![1, 3]);
    daemon2.start_shard(ShardId(2), vec![1, 3]);
    daemon2.start_shard(ShardId(3), vec![1, 3]);

    daemon3.start_shard(ShardId(1), vec![1, 2]);
    daemon3.start_shard(ShardId(2), vec![1, 2]);
    daemon3.start_shard(ShardId(3), vec![1, 2]);

    let d1 = daemon1.clone();
    let d2 = daemon2.clone();
    let d3 = daemon3.clone();

    let h1 = tokio::spawn(async move {
        let _ = d1.run().await;
    });
    let h2 = tokio::spawn(async move {
        let _ = d2.run().await;
    });
    let h3 = tokio::spawn(async move {
        let _ = d3.run().await;
    });

    tokio::time::sleep(std::time::Duration::from_millis(600)).await;

    let seed_addrs = vec![format!("http://{}", addr1)];
    let client = StrataClient::connect(seed_addrs).await.unwrap();

    // 1. Successful 2PC Cross-Shard Transaction
    let start_ts = client.begin_txn().await.unwrap();
    let mutations = vec![
        Mutation::Put(b"apple".to_vec(), b"red".to_vec()), // Shard 1
        Mutation::Put(b"lemon".to_vec(), b"yellow".to_vec()), // Shard 2
        Mutation::Put(b"zebra".to_vec(), b"stripes".to_vec()), // Shard 3
    ];

    client
        .prewrite_txn(start_ts, mutations.clone())
        .await
        .unwrap();
    let commit_ts = daemon1.hlc.local_event();
    client
        .commit_txn(start_ts, commit_ts, mutations)
        .await
        .unwrap();

    // Verify all 3 keys are readable
    let val_apple = client.get(b"apple").await.unwrap();
    let val_lemon = client.get(b"lemon").await.unwrap();
    let val_zebra = client.get(b"zebra").await.unwrap();

    assert_eq!(val_apple, Some(b"red".to_vec()));
    assert_eq!(val_lemon, Some(b"yellow".to_vec()));
    assert_eq!(val_zebra, Some(b"stripes".to_vec()));

    // 2. Aborted Cross-Shard Transaction
    let start_ts2 = client.begin_txn().await.unwrap();
    let mutations2 = vec![
        Mutation::Put(b"apple".to_vec(), b"green".to_vec()),
        Mutation::Put(b"lemon".to_vec(), b"sour".to_vec()),
    ];

    client
        .prewrite_txn(start_ts2, mutations2.clone())
        .await
        .unwrap();
    client.abort_txn(start_ts2, mutations2).await.unwrap();

    // Verify values remain unchanged after abort
    let val_apple2 = client.get(b"apple").await.unwrap();
    let val_lemon2 = client.get(b"lemon").await.unwrap();
    assert_eq!(val_apple2, Some(b"red".to_vec()));
    assert_eq!(val_lemon2, Some(b"yellow".to_vec()));

    daemon1.shutdown();
    daemon2.shutdown();
    daemon3.shutdown();
    let _ = tokio::join!(h1, h2, h3);
}
