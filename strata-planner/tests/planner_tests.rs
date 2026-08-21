use rand::Rng;
use roaring::RoaringBitmap;
use std::time::Instant;

use strata_index::hnsw::{HnswConfig, HnswIndex};
use strata_index::AnnIndex;
use strata_planner::{ExecutionStrategy, PlannerConfig, QueryPlanner, ShardStats, VectorQuery};

fn generate_dataset(n: usize, dim: usize) -> Vec<(u64, Vec<f32>)> {
    let mut rng = rand::thread_rng();
    (0..n as u64)
        .map(|id| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            (id, v)
        })
        .collect()
}

fn create_bitmap_with_selectivity(n: usize, selectivity: f64) -> RoaringBitmap {
    let mut bitmap = RoaringBitmap::new();
    let count = ((n as f64) * selectivity).round() as usize;
    for id in 0..count {
        bitmap.insert(id as u32);
    }
    bitmap
}

#[test]
fn test_planner_picks_bruteforce_below_threshold() {
    let planner = QueryPlanner::new();
    let query = VectorQuery::new(vec![0.1; 16], 10);

    // Below threshold (100 <= 500)
    let stats_small = ShardStats::new(100, 1024 * 100, false);
    let strategy_small = planner.plan(&query, &stats_small);
    assert_eq!(
        strategy_small,
        ExecutionStrategy::BruteForceScan,
        "Planner should select BruteForceScan for small datasets below threshold"
    );

    // Exactly at threshold boundary
    let stats_boundary = ShardStats::new(500, 1024 * 500, false);
    let strategy_boundary = planner.plan(&query, &stats_boundary);
    assert_eq!(
        strategy_boundary,
        ExecutionStrategy::BruteForceScan,
        "Planner should select BruteForceScan at threshold boundary"
    );

    // Above threshold (1000 > 500)
    let stats_large = ShardStats::new(1000, 1024 * 1000, false);
    let strategy_large = planner.plan(&query, &stats_large);
    assert_eq!(
        strategy_large,
        ExecutionStrategy::InMemoryHnsw,
        "Planner should select InMemoryHnsw for datasets above threshold"
    );
}

#[test]
fn test_planner_falls_back_to_disk_index_above_memory_threshold() {
    let config = PlannerConfig {
        memory_threshold_bytes: 10 * 1024 * 1024, // 10 MB threshold
        ..Default::default()
    };
    let planner = QueryPlanner::with_config(config);
    let query = VectorQuery::new(vec![0.1; 16], 10);

    // Below memory threshold (5 MB)
    let stats_in_mem = ShardStats::new(5000, 5 * 1024 * 1024, true);
    let strat_in_mem = planner.plan(&query, &stats_in_mem);
    assert_eq!(
        strat_in_mem,
        ExecutionStrategy::InMemoryHnsw,
        "Planner should remain in-memory if size is under memory threshold"
    );

    // Above memory threshold (15 MB) with disk index available
    let stats_disk = ShardStats::new(50000, 15 * 1024 * 1024, true);
    let strat_disk = planner.plan(&query, &stats_disk);
    assert_eq!(
        strat_disk,
        ExecutionStrategy::OutofCoreVamana,
        "Planner should fall back to disk index above memory threshold"
    );

    // Above memory threshold (15 MB) but no disk index available
    let stats_no_disk = ShardStats::new(50000, 15 * 1024 * 1024, false);
    let strat_no_disk = planner.plan(&query, &stats_no_disk);
    assert_eq!(
        strat_no_disk,
        ExecutionStrategy::InMemoryHnsw,
        "Planner should stay in-memory if no disk index exists even if memory exceeds threshold"
    );
}

#[test]
fn test_planner_picks_prefilter_vs_postfilter_by_selectivity() {
    let mut rng = rand::thread_rng();
    let dim = 32;
    let n = 2000;
    let k = 10;
    let dataset = generate_dataset(n, dim);

    let mut hnsw = HnswIndex::new(dim, HnswConfig::new(16, 100, 50));
    for (id, vec) in &dataset {
        hnsw.insert(*id, vec).unwrap();
    }

    let config = PlannerConfig {
        bruteforce_threshold: 100,
        memory_threshold_bytes: 100 * 1024 * 1024,
        prefilter_selectivity_threshold: 0.25,
    };
    let planner = QueryPlanner::with_config(config);

    // ── Low Selectivity Test (2% selectivity) ─────────────────────────────────
    let low_sel_bitmap = create_bitmap_with_selectivity(n, 0.02);
    let query_vec: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let low_sel_query = VectorQuery::new(query_vec.clone(), k).with_filter(low_sel_bitmap);
    let stats = ShardStats::new(n, 5 * 1024 * 1024, false);

    let chosen_strategy_low = planner.plan(&low_sel_query, &stats);
    assert_eq!(
        chosen_strategy_low,
        ExecutionStrategy::PrefilterGraph,
        "Planner should pick PrefilterGraph for low selectivity (2%)"
    );

    // Empirical benchmark comparison at low selectivity
    let num_queries = 20;
    let mut prefilter_time = 0u128;
    let mut postfilter_time = 0u128;

    for _ in 0..num_queries {
        let q: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let query = VectorQuery::new(q, k).with_filter(create_bitmap_with_selectivity(n, 0.02));

        let t0 = Instant::now();
        let pre_res = planner.execute_prefilter_hnsw(&hnsw, &query).unwrap();
        prefilter_time += t0.elapsed().as_nanos();

        let t1 = Instant::now();
        let post_res = planner
            .execute_postfilter_hnsw(&hnsw, &query, &stats)
            .unwrap();
        postfilter_time += t1.elapsed().as_nanos();

        // Verify pre-filter returns requested results if enough matches exist
        assert!(
            !pre_res.is_empty(),
            "Prefilter should return non-empty result"
        );
        let _ = post_res;
    }

    println!(
        "[Low Selectivity 2%] Prefilter avg: {} µs | Postfilter avg: {} µs",
        prefilter_time / (num_queries * 1000),
        postfilter_time / (num_queries * 1000)
    );

    // ── High Selectivity Test (90% selectivity) ──────────────────────────────
    let high_sel_bitmap = create_bitmap_with_selectivity(n, 0.90);
    let high_sel_query = VectorQuery::new(query_vec, k).with_filter(high_sel_bitmap);

    let chosen_strategy_high = planner.plan(&high_sel_query, &stats);
    assert_eq!(
        chosen_strategy_high,
        ExecutionStrategy::PostfilterGraph,
        "Planner should pick PostfilterGraph for high selectivity (90%)"
    );

    let mut prefilter_high_time = 0u128;
    let mut postfilter_high_time = 0u128;

    for _ in 0..num_queries {
        let q: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let query = VectorQuery::new(q, k).with_filter(create_bitmap_with_selectivity(n, 0.90));

        let t0 = Instant::now();
        let _ = planner.execute_prefilter_hnsw(&hnsw, &query).unwrap();
        prefilter_high_time += t0.elapsed().as_nanos();

        let t1 = Instant::now();
        let _ = planner
            .execute_postfilter_hnsw(&hnsw, &query, &stats)
            .unwrap();
        postfilter_high_time += t1.elapsed().as_nanos();
    }

    println!(
        "[High Selectivity 90%] Prefilter avg: {} µs | Postfilter avg: {} µs",
        prefilter_high_time / (num_queries * 1000),
        postfilter_high_time / (num_queries * 1000)
    );
}

#[test]
fn test_planner_benchmark_report_grid() {
    let dim = 32;
    let k = 10;
    let mut rng = rand::thread_rng();
    let planner = QueryPlanner::new();

    let sizes = vec![100usize, 1000usize, 5000usize];
    let selectivities = vec![0.01, 0.10, 0.50, 0.90];

    println!("\n==========================================================================================================");
    println!("                                PLANNER BENCHMARK REPORT GRID                                             ");
    println!("==========================================================================================================");
    println!(
        "{:<12} | {:<12} | {:<22} | {:<18} | {:<18} | {:<20}",
        "Dataset Size",
        "Selectivity",
        "Planner Strategy",
        "Planner Lat (µs)",
        "Always-HNSW (µs)",
        "Always-Brute (µs)"
    );
    println!("----------------------------------------------------------------------------------------------------------");

    for &n in &sizes {
        let dataset = generate_dataset(n, dim);
        let mut hnsw = HnswIndex::new(dim, HnswConfig::new(16, 100, 50));
        for (id, vec) in &dataset {
            hnsw.insert(*id, vec).unwrap();
        }
        let stats = ShardStats::new(n, n * dim * 4, false);

        for &sel in &selectivities {
            let bitmap = create_bitmap_with_selectivity(n, sel);
            let q_vec: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            let query = VectorQuery::new(q_vec.clone(), k).with_filter(bitmap.clone());

            let chosen_strategy = planner.plan(&query, &stats);

            let iterations = 10;

            // 1. Planner Strategy Latency
            let t0 = Instant::now();
            for _ in 0..iterations {
                match chosen_strategy {
                    ExecutionStrategy::BruteForceScan => {
                        let _ = planner.execute_bruteforce(&dataset, &query);
                    }
                    ExecutionStrategy::PrefilterGraph => {
                        let _ = planner.execute_prefilter_hnsw(&hnsw, &query);
                    }
                    ExecutionStrategy::PostfilterGraph => {
                        let _ = planner.execute_postfilter_hnsw(&hnsw, &query, &stats);
                    }
                    _ => {
                        let _ = planner.execute_prefilter_hnsw(&hnsw, &query);
                    }
                }
            }
            let planner_lat = t0.elapsed().as_micros() as f64 / iterations as f64;

            // 2. Always-HNSW Latency (Post-filter / direct)
            let t1 = Instant::now();
            for _ in 0..iterations {
                let _ = planner.execute_postfilter_hnsw(&hnsw, &query, &stats);
            }
            let always_hnsw_lat = t1.elapsed().as_micros() as f64 / iterations as f64;

            // 3. Always-Bruteforce Latency
            let t2 = Instant::now();
            for _ in 0..iterations {
                let _ = planner.execute_bruteforce(&dataset, &query);
            }
            let always_brute_lat = t2.elapsed().as_micros() as f64 / iterations as f64;

            println!(
                "{:<12} | {:<12.2} | {:<22?} | {:<18.2} | {:<18.2} | {:<20.2}",
                n, sel, chosen_strategy, planner_lat, always_hnsw_lat, always_brute_lat
            );
        }
    }
    println!("==========================================================================================================\n");
}
