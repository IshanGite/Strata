use rand::Rng;
use serde::{Deserialize, Serialize};

// ----------------------------------------------------------------------
// 1. Distance Metrics: Scalar Module
// ----------------------------------------------------------------------
pub mod scalar {
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }

    pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let dot = dot_product(a, b);
        let norm_a = dot_product(a, a).sqrt();
        let norm_b = dot_product(b, b).sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            1.0
        } else {
            let val = 1.0 - (dot / (norm_a * norm_b));
            if val < 0.0 {
                0.0
            } else {
                val
            }
        }
    }

    pub fn l1_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
    }
}

// ----------------------------------------------------------------------
// 2. Distance Metrics: NEON Module
// ----------------------------------------------------------------------
#[cfg(target_arch = "aarch64")]
pub mod neon {
    use std::arch::aarch64::*;

    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        unsafe {
            let mut sum_vec = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 3 < len {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                sum_vec = vaddq_f32(sum_vec, vmulq_f32(va, vb));
                i += 4;
            }
            let mut sum = vaddvq_f32(sum_vec);
            while i < len {
                sum += a[i] * b[i];
                i += 1;
            }
            sum
        }
    }

    pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        unsafe {
            let mut sum_vec = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 3 < len {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                let diff = vsubq_f32(va, vb);
                sum_vec = vaddq_f32(sum_vec, vmulq_f32(diff, diff));
                i += 4;
            }
            let mut sum = vaddvq_f32(sum_vec);
            while i < len {
                let diff = a[i] - b[i];
                sum += diff * diff;
                i += 1;
            }
            sum.sqrt()
        }
    }

    pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let dot = dot_product(a, b);
        let norm_a = dot_product(a, a).sqrt();
        let norm_b = dot_product(b, b).sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            1.0
        } else {
            let val = 1.0 - (dot / (norm_a * norm_b));
            if val < 0.0 {
                0.0
            } else {
                val
            }
        }
    }

    pub fn l1_distance(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        unsafe {
            let mut sum_vec = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 3 < len {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                let diff = vsubq_f32(va, vb);
                sum_vec = vaddq_f32(sum_vec, vabsq_f32(diff));
                i += 4;
            }
            let mut sum = vaddvq_f32(sum_vec);
            while i < len {
                sum += (a[i] - b[i]).abs();
                i += 1;
            }
            sum
        }
    }
}

// ----------------------------------------------------------------------
// 3. Dispatch / Fallback Wrappers
// ----------------------------------------------------------------------
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::dot_product(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar::dot_product(a, b)
    }
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::l2_distance(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar::l2_distance(a, b)
    }
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::cosine_distance(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar::cosine_distance(a, b)
    }
}

pub fn l1_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        neon::l1_distance(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar::l1_distance(a, b)
    }
}

// ----------------------------------------------------------------------
// 4. K-Means Clustering implementation (Lloyd's + k-means++)
// ----------------------------------------------------------------------
pub fn train_kmeans(data: &[Vec<f32>], k: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    if data.is_empty() {
        return vec![vec![0.0; dim]; k];
    }

    // k-means++ Initialization
    let mut centroids = Vec::with_capacity(k);
    let first_idx = rng.gen_range(0..data.len());
    centroids.push(data[first_idx].clone());

    let mut min_dists = vec![f32::MAX; data.len()];
    for _ in 1..k {
        let last_c = &centroids[centroids.len() - 1];
        let mut total_dist = 0.0;
        for (i, p) in data.iter().enumerate() {
            let d = scalar::l2_distance(p, last_c);
            let d2 = d * d;
            if d2 < min_dists[i] {
                min_dists[i] = d2;
            }
            total_dist += min_dists[i];
        }

        if total_dist <= 0.0 {
            // Duplicate remaining randomly
            let next_idx = rng.gen_range(0..data.len());
            centroids.push(data[next_idx].clone());
            continue;
        }

        let mut target = rng.gen::<f32>() * total_dist;
        let mut selected = 0;
        for (i, &d2) in min_dists.iter().enumerate() {
            target -= d2;
            if target <= 0.0 {
                selected = i;
                break;
            }
        }
        centroids.push(data[selected].clone());
    }

    // Lloyd's iterations
    let max_iters = 15;
    for _ in 0..max_iters {
        let mut assignments = vec![0; data.len()];
        for (i, p) in data.iter().enumerate() {
            let mut best_c = 0;
            let mut best_d = f32::MAX;
            for (c_idx, c) in centroids.iter().enumerate() {
                let d = scalar::l2_distance(p, c);
                if d < best_d {
                    best_d = d;
                    best_c = c_idx;
                }
            }
            assignments[i] = best_c;
        }

        let mut new_sums = vec![vec![0.0; dim]; k];
        let mut counts = vec![0; k];
        for (i, &c_idx) in assignments.iter().enumerate() {
            counts[c_idx] += 1;
            for d in 0..dim {
                new_sums[c_idx][d] += data[i][d];
            }
        }

        let mut shift = 0.0;
        for c_idx in 0..k {
            if counts[c_idx] > 0 {
                let mut new_c = vec![0.0; dim];
                for d in 0..dim {
                    new_c[d] = new_sums[c_idx][d] / counts[c_idx] as f32;
                }
                shift += scalar::l2_distance(&centroids[c_idx], &new_c);
                centroids[c_idx] = new_c;
            } else {
                // Empty centroid: reinitialize to random data point
                let rand_idx = rng.gen_range(0..data.len());
                centroids[c_idx] = data[rand_idx].clone();
            }
        }

        if shift < 1e-4 {
            break;
        }
    }

    centroids
}

// ----------------------------------------------------------------------
// 5. Product Quantization (PQ) Support
// ----------------------------------------------------------------------
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProductQuantizer {
    pub m: usize,
    pub d_sub: usize,
    pub codebooks: Vec<Vec<Vec<f32>>>, // [subspace][centroid_idx][dim]
}

impl ProductQuantizer {
    pub fn train(dataset: &[&[f32]], m: usize) -> Self {
        if dataset.is_empty() {
            return Self {
                m,
                d_sub: 0,
                codebooks: Vec::new(),
            };
        }
        let total_dim = dataset[0].len();
        let d_sub = total_dim / m;
        assert_eq!(
            total_dim % m,
            0,
            "Dimension must be divisible by subspaces M"
        );

        let mut codebooks = Vec::with_capacity(m);

        for s in 0..m {
            let mut sub_vectors = Vec::with_capacity(dataset.len());
            for &vec in dataset {
                sub_vectors.push(vec[s * d_sub..(s + 1) * d_sub].to_vec());
            }
            let centroids = train_kmeans(&sub_vectors, 256, d_sub);
            codebooks.push(centroids);
        }

        Self {
            m,
            d_sub,
            codebooks,
        }
    }

    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut encoded = Vec::with_capacity(self.m);
        for s in 0..self.m {
            let sub = &vector[s * self.d_sub..(s + 1) * self.d_sub];
            let mut best_idx = 0;
            let mut best_d = f32::MAX;
            for (c_idx, centroid) in self.codebooks[s].iter().enumerate() {
                let d = scalar::l2_distance(sub, centroid);
                if d < best_d {
                    best_d = d;
                    best_idx = c_idx;
                }
            }
            encoded.push(best_idx as u8);
        }
        encoded
    }

    pub fn decode(&self, encoded: &[u8]) -> Vec<f32> {
        let mut decoded = Vec::with_capacity(self.m * self.d_sub);
        for (s, &byte) in encoded.iter().enumerate() {
            decoded.extend_from_slice(&self.codebooks[s][byte as usize]);
        }
        decoded
    }

    pub fn distance_table(&self, query: &[f32]) -> Vec<f32> {
        let mut table = vec![0.0; self.m * 256];
        for s in 0..self.m {
            let sub = &query[s * self.d_sub..(s + 1) * self.d_sub];
            for k in 0..256 {
                table[s * 256 + k] = scalar::l2_distance(sub, &self.codebooks[s][k]);
            }
        }
        table
    }

    pub fn adc_distance(&self, lookups: &[f32], encoded: &[u8]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            unsafe {
                let m = encoded.len();
                let mut sum_vec = vdupq_n_f32(0.0);

                // Process in chunks of 4 subspaces
                let mut i = 0;
                while i + 3 < m {
                    let d0 = lookups[i * 256 + encoded[i] as usize];
                    let d1 = lookups[(i + 1) * 256 + encoded[i + 1] as usize];
                    let d2 = lookups[(i + 2) * 256 + encoded[i + 2] as usize];
                    let d3 = lookups[(i + 3) * 256 + encoded[i + 3] as usize];

                    let vals = [d0, d1, d2, d3];
                    let chunk = vld1q_f32(vals.as_ptr());
                    sum_vec = vaddq_f32(sum_vec, chunk);
                    i += 4;
                }

                let mut sum = vaddvq_f32(sum_vec);

                // Process remaining subspaces
                while i < m {
                    sum += lookups[i * 256 + encoded[i] as usize];
                    i += 1;
                }
                sum
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut sum = 0.0;
            for (i, &byte) in encoded.iter().enumerate() {
                sum += lookups[i * 256 + byte as usize];
            }
            sum
        }
    }
}
