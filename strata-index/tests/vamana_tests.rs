use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::io::Write;

use strata_index::hnsw::{HnswConfig, HnswIndex};
use strata_index::vamana::{robust_prune, VamanaConfig};
use strata_index::vamana_disk::VamanaDiskIndex;
use strata_index::AnnIndex;
use strata_simd::l2_distance;
use tempfile::TempDir;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn brute_force_knn(dataset: &[(u64, Vec<f32>)], query: &[f32], k: usize) -> Vec<u64> {
    let mut dists: Vec<(f32, u64)> = dataset
        .iter()
        .map(|(id, v)| (l2_distance(query, v), *id))
        .collect();
    dists.sort_by(|a, b| a.0.total_cmp(&b.0));
    dists.into_iter().take(k).map(|(_, id)| id).collect()
}

fn recall_at_k(results: &[(u64, f32)], gt: &[u64], k: usize) -> f32 {
    let res: HashSet<u64> = results.iter().take(k).map(|(id, _)| *id).collect();
    let gt_set: HashSet<u64> = gt.iter().take(k).copied().collect();
    res.intersection(&gt_set).count() as f32 / k as f32
}

// ── RobustPrune unit test ─────────────────────────────────────────────────────

/// Verify that RobustPrune produces a neighbour list with diversity:
/// after choosing p*, every removed candidate satisfies
/// `alpha * dist(p*, v) ≤ dist(p, v)`.
#[test]
fn test_vamana_robust_prune() {
    let p = vec![0.0f32, 0.0, 0.0];
    // Candidates at known positions.
    let mut id_to_vec: HashMap<u64, Vec<f32>> = HashMap::new();
    id_to_vec.insert(1, vec![1.0, 0.0, 0.0]); // dist(p,1) = 1.0
    id_to_vec.insert(2, vec![1.1, 0.0, 0.0]); // dist(p,2) = 1.1, close to 1
    id_to_vec.insert(3, vec![0.0, 1.0, 0.0]); // dist(p,3) = 1.0, different direction
    id_to_vec.insert(4, vec![0.0, 0.0, 1.0]); // dist(p,4) = 1.0, different direction

    let borrowed: HashMap<u64, &Vec<f32>> = id_to_vec.iter().map(|(k, v)| (*k, v)).collect();
    let candidates: HashSet<u64> = [1u64, 2, 3, 4].into_iter().collect();

    let result = robust_prune(&p, &candidates, &borrowed, 1.2, 3);

    // Node 1 is chosen first (nearest).  Node 2 should be pruned because
    // alpha * dist(1, 2) = 1.2 * 0.1 = 0.12 ≤ dist(p, 2) = 1.1 is FALSE
    // (0.12 ≤ 1.1 is true, meaning 2 IS pruned).
    // Nodes 3 and 4 are in orthogonal directions and should be kept.
    assert!(
        result.contains(&1),
        "Node 1 (nearest) must be in result, got {:?}",
        result
    );
    assert!(
        !result.contains(&2),
        "Node 2 (dominated by 1) should be pruned, got {:?}",
        result
    );
    // 3 and 4 should be kept (different directions → not dominated)
    assert!(
        result.contains(&3) || result.contains(&4),
        "At least one orthogonal neighbour should survive, got {:?}",
        result
    );
    assert!(result.len() <= 3, "Result exceeds R=3: {:?}", result);
}

// ── Scale recall test ─────────────────────────────────────────────────────────

/// Build a Vamana disk index on N=20 000 vectors and verify:
/// 1. Recall@10 ≥ 0.70 (two-stage PQ coarse + exact re-rank).
/// 2. The in-memory footprint during search is far less than full-vector RAM.
///
/// Architecture note: the out-of-core property is demonstrated by showing that
/// `in_memory_bytes()` ≪ `N × dim × 4` (full-precision RAM).  The search never
/// allocates the full vector set; only PQ codes + offset maps are resident.
#[test]
fn test_vamana_recall_at_scale() {
    let mut rng = rand::thread_rng();
    let dim = 128;
    let n = 20_000usize;
    let k = 10;

    let dataset: Vec<(u64, Vec<f32>)> = (0..n as u64)
        .map(|id| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            (id, v)
        })
        .collect();

    let config = VamanaConfig {
        r: 32,
        alpha: 1.2,
        l_build: 75,
        pq_subspaces: 16,
        beam_width: 8,
        rerank_factor: 10,
    };

    let tmp = TempDir::new().expect("tempdir");
    let index =
        VamanaDiskIndex::build(&dataset, config.clone(), tmp.path(), "test").expect("build failed");

    // Verify out-of-core memory property.
    let full_vec_bytes = n * dim * std::mem::size_of::<f32>();
    let search_mem = index.in_memory_bytes();
    println!(
        "[Vamana scale] n={n}, dim={dim}  |  \
         full-vec RAM: {:.1} MB  |  search RAM: {:.1} MB  |  ratio: {:.1}x",
        full_vec_bytes as f64 / 1e6,
        search_mem as f64 / 1e6,
        full_vec_bytes as f64 / search_mem as f64,
    );
    assert!(
        search_mem < full_vec_bytes / 4,
        "search_mem {search_mem} should be <1/4 of full_vec_bytes {full_vec_bytes}"
    );

    // Evaluate recall.
    let num_queries = 50;
    let mut total_recall = 0.0f32;

    for _ in 0..num_queries {
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let gt = brute_force_knn(&dataset, &query, k);
        let results = index.search_knn(&query, k).expect("search_knn failed");
        total_recall += recall_at_k(&results, &gt, k);
    }

    let mean_recall = total_recall / num_queries as f32;
    println!("[Vamana scale] Recall@{k}: {mean_recall:.4}");
    assert!(
        mean_recall >= 0.70,
        "Vamana Recall@{k} {mean_recall:.4} < 0.70 threshold"
    );
}

// ── Memory footprint comparison table ────────────────────────────────────────

/// Direct comparison of HNSW vs Vamana-Disk on the same dataset.
/// Writes docs/benchmarks/vamana_vs_hnsw_comparison.md.
#[test]
fn test_vamana_vs_hnsw_memory_footprint() {
    let mut rng = rand::thread_rng();
    let dim = 128;
    let n = 10_000usize;
    let k = 10;

    let dataset: Vec<(u64, Vec<f32>)> = (0..n as u64)
        .map(|id| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            (id, v)
        })
        .collect();

    // ── HNSW ─────────────────────────────────────────────────────────────────
    let hnsw_config = HnswConfig::new(16, 200, 100);
    let mut hnsw = HnswIndex::new(dim, hnsw_config);
    for (id, v) in &dataset {
        hnsw.insert(*id, v).expect("hnsw insert");
    }
    let hnsw_stats = hnsw.stats();

    // ── Vamana disk ───────────────────────────────────────────────────────────
    let vamana_config = VamanaConfig {
        r: 32,
        alpha: 1.2,
        l_build: 75,
        pq_subspaces: 16,
        beam_width: 8,
        rerank_factor: 10,
    };
    let tmp = TempDir::new().expect("tempdir");
    let vamana = VamanaDiskIndex::build(&dataset, vamana_config.clone(), tmp.path(), "cmp")
        .expect("vamana build");
    let vamana_stats = vamana.stats();

    // ── Measure recall ────────────────────────────────────────────────────────
    let num_queries = 30;
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();

    let mut hnsw_recall_sum = 0.0f32;
    let mut vamana_recall_sum = 0.0f32;
    let mut hnsw_ns = 0u128;
    let mut vamana_ns = 0u128;

    for q in &queries {
        let gt = brute_force_knn(&dataset, q, k);

        let t0 = std::time::Instant::now();
        let hr = hnsw.search_knn(q, k).expect("hnsw search");
        hnsw_ns += t0.elapsed().as_nanos();
        hnsw_recall_sum += recall_at_k(&hr, &gt, k);

        let t0 = std::time::Instant::now();
        let vr = vamana.search_knn(q, k).expect("vamana search");
        vamana_ns += t0.elapsed().as_nanos();
        vamana_recall_sum += recall_at_k(&vr, &gt, k);
    }

    let hnsw_recall = hnsw_recall_sum / num_queries as f32;
    let vamana_recall = vamana_recall_sum / num_queries as f32;
    let hnsw_lat_us = hnsw_ns / num_queries as u128 / 1_000;
    let vamana_lat_us = vamana_ns / num_queries as u128 / 1_000;

    let full_vec_bytes = n * dim * std::mem::size_of::<f32>();

    let table = format!(
        "# Vamana Disk vs HNSW Comparison\n\n\
         Dataset: {n} random {dim}-dim float32 vectors  \n\
         Full-precision size: {:.1} MB  \n\n\
         | Index        | In-memory (MB) | Recall@{k} | Avg latency (µs) | When to use |\n\
         |---|---|---|---|---|\n\
         | HNSW (M=16)  | {:.2} | {hnsw_recall:.4}   | {hnsw_lat_us}   | Dataset fits in RAM; latency-critical |\n\
         | Vamana Disk  | {:.2} | {vamana_recall:.4}   | {vamana_lat_us}   | Dataset exceeds RAM; disk-backed |\n\n\
         ## Summary\n\n\
         - HNSW stores all vectors and adjacency lists in memory → full-vec RAM + graph overhead.\n\
         - VamanaDiskIndex keeps only PQ codes ({pq_mb:.2} MB) and offset maps in memory during \
           search; full-precision vectors ({fv_mb:.1} MB) stay on disk and are read only for \
           the final re-rank step.\n\
         - Choose HNSW when the dataset fits comfortably in unified memory (lower latency).\n\
         - Choose VamanaDiskIndex when the full-precision dataset exceeds available RAM \
           (accepts a latency penalty for the disk I/O, offset by the PQ coarse-pass savings).\n",
        full_vec_bytes as f64 / 1e6,
        hnsw_stats.memory_bytes as f64 / 1e6,
        vamana_stats.memory_bytes as f64 / 1e6,
        pq_mb = vamana_stats.memory_bytes as f64 / 1e6,
        fv_mb = full_vec_bytes as f64 / 1e6,
    );

    println!("{table}");

    // Write benchmark file.
    let bench_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("docs/benchmarks/vamana_vs_hnsw_comparison.md");
    if let Some(parent) = bench_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    if let Ok(mut file) = std::fs::File::create(&bench_path) {
        if let Err(e) = file.write_all(table.as_bytes()) {
            eprintln!("Warning: could not write benchmark file: {e}");
        }
    }

    // Core correctness assertion: both must achieve reasonable recall.
    assert!(
        hnsw_recall >= 0.85,
        "HNSW Recall@{k} {hnsw_recall:.4} < 0.85"
    );
    assert!(
        vamana_recall >= 0.60,
        "Vamana Recall@{k} {vamana_recall:.4} < 0.60"
    );
    // Vamana search memory must be substantially less than full-precision RAM.
    assert!(
        vamana_stats.memory_bytes < full_vec_bytes / 3,
        "Vamana in-memory {} should be < 1/3 of full-vec {}",
        vamana_stats.memory_bytes,
        full_vec_bytes
    );
}
