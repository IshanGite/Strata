use proptest::prelude::*;
use rand::Rng;
#[cfg(target_arch = "aarch64")]
use strata_simd::neon;
use strata_simd::{
    cosine_distance, dot_product, l1_distance, l2_distance, scalar, ProductQuantizer,
};

// 1. Proptest Equivalence Test for Distance Functions
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]
    #[test]
    fn test_simd_scalar_equivalence(
        dim in 1..1000usize,
        seed_a in any::<u64>(),
        seed_b in any::<u64>(),
    ) {
        use rand::SeedableRng;
        let mut rng_a = rand::rngs::StdRng::seed_from_u64(seed_a);
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(seed_b);

        let a: Vec<f32> = (0..dim).map(|_| rng_a.gen::<f32>() * 10.0 - 5.0).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng_b.gen::<f32>() * 10.0 - 5.0).collect();

        // 1. Dot Product
        let dot_scalar = scalar::dot_product(&a, &b);
        let dot_dispatch = dot_product(&a, &b);
        assert!((dot_scalar - dot_dispatch).abs() < 1e-2);

        // 2. L2 Distance
        let l2_scalar = scalar::l2_distance(&a, &b);
        let l2_dispatch = l2_distance(&a, &b);
        assert!((l2_scalar - l2_dispatch).abs() < 1e-2);

        // 3. Cosine Distance
        let cos_scalar = scalar::cosine_distance(&a, &b);
        let cos_dispatch = cosine_distance(&a, &b);
        assert!((cos_scalar - cos_dispatch).abs() < 1e-2);

        // 4. L1 Distance
        let l1_scalar = scalar::l1_distance(&a, &b);
        let l1_dispatch = l1_distance(&a, &b);
        assert!((l1_scalar - l1_dispatch).abs() < 1e-2);

        #[cfg(target_arch = "aarch64")]
        {
            // Verify NEON matches scalar directly
            let dot_neon = neon::dot_product(&a, &b);
            let l2_neon = neon::l2_distance(&a, &b);
            let cos_neon = neon::cosine_distance(&a, &b);
            let l1_neon = neon::l1_distance(&a, &b);

            assert!((dot_scalar - dot_neon).abs() < 1e-2);
            assert!((l2_scalar - l2_neon).abs() < 1e-2);
            assert!((cos_scalar - cos_neon).abs() < 1e-2);
            assert!((l1_scalar - l1_neon).abs() < 1e-2);
        }
    }
}

// 2. PQ Encode/Decode Roundtrip Error Bounds
#[test]
fn test_pq_encode_decode_roundtrip_error_bounds() {
    let mut rng = rand::thread_rng();
    let dim = 128;
    let n = 500;

    // Generate random float vectors
    let mut dataset_vecs = Vec::with_capacity(n);
    for _ in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        dataset_vecs.push(v);
    }
    let dataset: Vec<&[f32]> = dataset_vecs.iter().map(|v| v.as_slice()).collect();

    for m in [8, 16] {
        let pq = ProductQuantizer::train(&dataset, m);
        let test_vec = &dataset_vecs[0];
        let encoded = pq.encode(test_vec);
        assert_eq!(encoded.len(), m);

        let decoded = pq.decode(&encoded);
        assert_eq!(decoded.len(), dim);

        // Calculate reconstruction error (RMSE)
        let rmse = l2_distance(test_vec, &decoded) / (dim as f32).sqrt();
        assert!(rmse < 0.5, "RMSE for M={} exceeds error bound: {}", m, rmse);
    }
}

// 3. ADC Matches Exact Distance Ranking Correlation
#[test]
fn test_adc_matches_exact_distance_ranking() {
    let mut rng = rand::thread_rng();
    let dim = 128;
    let n = 200;
    let m = 16;

    let mut dataset_vecs = Vec::with_capacity(n);
    for _ in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        dataset_vecs.push(v);
    }
    let dataset: Vec<&[f32]> = dataset_vecs.iter().map(|v| v.as_slice()).collect();

    let pq = ProductQuantizer::train(&dataset, m);

    // Encode all vectors
    let encoded_dataset: Vec<Vec<u8>> = dataset_vecs.iter().map(|v| pq.encode(v)).collect();

    // Query vector
    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

    // Compute exact rankings
    let mut exact_distances: Vec<(usize, f32)> = dataset_vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2_distance(&query, v)))
        .collect();
    exact_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Compute ADC rankings
    let lookup_table = pq.distance_table(&query);
    let mut adc_distances: Vec<(usize, f32)> = encoded_dataset
        .iter()
        .enumerate()
        .map(|(i, enc)| (i, pq.adc_distance(&lookup_table, enc)))
        .collect();
    adc_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Quantify top-20 recall
    let k = 20;
    let exact_top_k: std::collections::HashSet<usize> = exact_distances
        .iter()
        .take(k)
        .map(|&(idx, _)| idx)
        .collect();

    let adc_top_k: std::collections::HashSet<usize> =
        adc_distances.iter().take(k).map(|&(idx, _)| idx).collect();

    let recall = exact_top_k.intersection(&adc_top_k).count() as f32 / k as f32;
    println!("ADC top-{} recall: {}", k, recall);

    // PQ is lossy but top-k correlation should be significant (>= 45%)
    assert!(recall >= 0.45, "Recall for M=16 was too low: {}", recall);
}

// 4. Benchmarking latency and recall curves
#[test]
fn test_run_benchmarks_and_quantize_curves() {
    let mut rng = rand::thread_rng();
    let dim = 128;
    let n = 1000;

    let mut dataset_vecs = Vec::with_capacity(n);
    for _ in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        dataset_vecs.push(v);
    }
    let dataset: Vec<&[f32]> = dataset_vecs.iter().map(|v| v.as_slice()).collect();

    // 1. Distance Metric Latency Benchmarks
    let test_a = &dataset_vecs[0];
    let test_b = &dataset_vecs[1];

    let iters = 10_000;

    // Dot Product Bench
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = scalar::dot_product(test_a, test_b);
    }
    let scalar_dot_time = start.elapsed().as_nanos() as f64 / iters as f64;

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = dot_product(test_a, test_b);
    }
    let dispatch_dot_time = start.elapsed().as_nanos() as f64 / iters as f64;

    // L2 Distance Bench
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = scalar::l2_distance(test_a, test_b);
    }
    let scalar_l2_time = start.elapsed().as_nanos() as f64 / iters as f64;

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = l2_distance(test_a, test_b);
    }
    let dispatch_l2_time = start.elapsed().as_nanos() as f64 / iters as f64;

    // Cosine Bench
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = scalar::cosine_distance(test_a, test_b);
    }
    let scalar_cos_time = start.elapsed().as_nanos() as f64 / iters as f64;

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = cosine_distance(test_a, test_b);
    }
    let dispatch_cos_time = start.elapsed().as_nanos() as f64 / iters as f64;

    // L1 Bench
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = scalar::l1_distance(test_a, test_b);
    }
    let scalar_l1_time = start.elapsed().as_nanos() as f64 / iters as f64;

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = l1_distance(test_a, test_b);
    }
    let dispatch_l1_time = start.elapsed().as_nanos() as f64 / iters as f64;

    // 2. PQ recall and compression latency benchmark curves
    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let mut exact_distances: Vec<(usize, f32)> = dataset_vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2_distance(&query, v)))
        .collect();
    exact_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let k = 20;
    let exact_top_k: std::collections::HashSet<usize> = exact_distances
        .iter()
        .take(k)
        .map(|&(idx, _)| idx)
        .collect();

    println!(
        "\n| Metric | Dimension | Scalar Latency (ns) | NEON / Dispatch Latency (ns) | Speedup |"
    );
    println!("|---|---|---|---|---|");
    println!(
        "| Dot Product | {} | {:.2} | {:.2} | {:.2}x |",
        dim,
        scalar_dot_time,
        dispatch_dot_time,
        scalar_dot_time / dispatch_dot_time
    );
    println!(
        "| L2 Distance | {} | {:.2} | {:.2} | {:.2}x |",
        dim,
        scalar_l2_time,
        dispatch_l2_time,
        scalar_l2_time / dispatch_l2_time
    );
    println!(
        "| Cosine Dist | {} | {:.2} | {:.2} | {:.2}x |",
        dim,
        scalar_cos_time,
        dispatch_cos_time,
        scalar_cos_time / dispatch_cos_time
    );
    println!(
        "| L1 Distance | {} | {:.2} | {:.2} | {:.2}x |",
        dim,
        scalar_l1_time,
        dispatch_l1_time,
        scalar_l1_time / dispatch_l1_time
    );

    println!(
        "\n| Compression Ratio | M Subspaces | Bytes per Vector | ADC Latency (ns) | Recall@{} |",
        k
    );
    println!("|---|---|---|---|---|");

    for m in [8, 16, 32] {
        let pq = ProductQuantizer::train(&dataset, m);
        let encoded: Vec<Vec<u8>> = dataset_vecs.iter().map(|v| pq.encode(v)).collect();
        let lookup_table = pq.distance_table(&query);

        let start = std::time::Instant::now();
        let mut adc_dists = Vec::with_capacity(n);
        for enc in &encoded {
            adc_dists.push(pq.adc_distance(&lookup_table, enc));
        }
        let batch_time = start.elapsed().as_nanos() as f64 / n as f64;

        let mut adc_distances: Vec<(usize, f32)> = adc_dists.into_iter().enumerate().collect();
        adc_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let adc_top_k: std::collections::HashSet<usize> =
            adc_distances.iter().take(k).map(|&(idx, _)| idx).collect();

        let recall = exact_top_k.intersection(&adc_top_k).count() as f32 / k as f32;
        println!(
            "| {:.2}x | {} | {} B | {:.2} | {:.2}% |",
            (dim * 4) as f32 / m as f32,
            m,
            m,
            batch_time,
            recall * 100.0
        );
    }
    println!();
}
