use rand::Rng;
use strata_index::hnsw::{HnswConfig, HnswIndex};
use strata_index::AnnIndex;
use strata_simd::l2_distance;

// ── Brute-force ground truth ──────────────────────────────────────────────────

fn brute_force_knn(dataset: &[(u64, Vec<f32>)], query: &[f32], k: usize) -> Vec<u64> {
    let mut dists: Vec<(f32, u64)> = dataset
        .iter()
        .map(|(id, v)| (l2_distance(query, v), *id))
        .collect();
    dists.sort_by(|a, b| a.0.total_cmp(&b.0));
    dists.into_iter().take(k).map(|(_, id)| id).collect()
}

// ── Recall helper ─────────────────────────────────────────────────────────────

fn recall_at_k(result_ids: &[(u64, f32)], ground_truth: &[u64], k: usize) -> f32 {
    use std::collections::HashSet;
    let res: HashSet<u64> = result_ids.iter().take(k).map(|(id, _)| *id).collect();
    let gt: HashSet<u64> = ground_truth.iter().take(k).copied().collect();
    res.intersection(&gt).count() as f32 / k as f32
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Build a 10 k-vector HNSW and verify Recall@10 ≥ 0.90 against brute-force.
#[test]
fn test_hnsw_recall_vs_bruteforce_ground_truth() {
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

    let config = HnswConfig::new(16, 200, 100);
    let mut index = HnswIndex::new(dim, config);
    for (id, vec) in &dataset {
        index.insert(*id, vec).expect("insert failed");
    }

    let num_queries = 100;
    let mut total_recall = 0.0f32;

    for _ in 0..num_queries {
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let gt = brute_force_knn(&dataset, &query, k);
        let results = index.search_knn(&query, k).expect("search_knn failed");
        total_recall += recall_at_k(&results, &gt, k);
    }

    let mean_recall = total_recall / num_queries as f32;
    println!("[HNSW recall] n={n}, M=16, ef_s=100, Recall@{k}: {mean_recall:.4}");
    assert!(
        mean_recall >= 0.90,
        "Recall@{k} {mean_recall:.4} < 0.90 threshold"
    );
}

/// Insert 1 000 vectors, delete 500 of them, then verify deleted IDs never
/// appear in search results.
#[test]
fn test_hnsw_insert_delete() {
    let mut rng = rand::thread_rng();
    let dim = 64;
    let n = 1_000usize;

    let mut index = HnswIndex::new(dim, HnswConfig::new(8, 50, 50));

    for id in 0..n as u64 {
        let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        index.insert(id, &v).expect("insert failed");
    }

    // Delete the upper half.
    for id in 500..n as u64 {
        index.delete(id).expect("delete failed");
    }

    for _ in 0..20 {
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let results = index.search_knn(&query, 10).expect("search_knn failed");
        for (id, _) in results {
            assert!(id < 500, "Deleted ID {id} appeared in search results");
        }
    }
}

/// Serialise → deserialise → confirm search results are identical.
#[test]
fn test_hnsw_serialization_roundtrip() {
    let mut rng = rand::thread_rng();
    let dim = 32;
    let mut index = HnswIndex::new(dim, HnswConfig::new(8, 50, 50));

    for id in 0..200u64 {
        let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        index.insert(id, &v).expect("insert failed");
    }

    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let before = index.search_knn(&query, 5).expect("search before");

    let bytes = index.to_bytes().expect("to_bytes failed");
    let index2 = HnswIndex::from_bytes(&bytes).expect("from_bytes failed");
    let after = index2.search_knn(&query, 5).expect("search after");

    assert_eq!(
        before.len(),
        after.len(),
        "Result count changed after roundtrip"
    );
    for ((id_b, d_b), (id_a, d_a)) in before.iter().zip(after.iter()) {
        assert_eq!(id_b, id_a, "Result ID changed after roundtrip");
        assert!(
            (d_b - d_a).abs() < 1e-5,
            "Distance changed after roundtrip: {d_b} vs {d_a}"
        );
    }
}

/// stats() reports consistent metadata.
#[test]
fn test_hnsw_stats() {
    let mut rng = rand::thread_rng();
    let dim = 16;
    let mut index = HnswIndex::new(dim, HnswConfig::new(4, 20, 20));
    for id in 0..50u64 {
        let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        index.insert(id, &v).expect("insert failed");
    }
    let stats = index.stats();
    assert_eq!(stats.num_vectors, 50);
    assert_eq!(stats.index_type, "hnsw");
    assert!(stats.memory_bytes > 0);
}
