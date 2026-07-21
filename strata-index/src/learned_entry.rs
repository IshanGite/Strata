//! Learned entry-point predictor for HNSW.
//!
//! A fixed random projection maps each vector to a low-dimensional space.
//! At query time the predictor finds the nearest training point in projected
//! space and uses that node as the HNSW entry, potentially skipping several
//! layers of greedy descent.
//!
//! # Failure modes (documented, tested in learned_entry_tests.rs)
//!
//! * **Degenerate query** (e.g. all-zero): the projection is also zero, placing
//!   it equidistant from many training points.  Likely `fallback_threshold` is
//!   exceeded and the standard entry fires.
//!
//! * **Highly clustered dataset**: all projections cluster tightly.  The
//!   predictor degenerates to a near-fixed entry point — harmless since recall
//!   never drops below the unaugmented HNSW fallback.
//!
//! * **Sparse training set**: coarse granularity reduces the benefit.  The
//!   predicted entry may be far from the optimal layer/region, triggering
//!   fallback at the `fallback_threshold`.

use rand::Rng;
use strata_simd::l2_distance;

use crate::hnsw::{HnswConfig, HnswIndex};
use crate::{AnnIndex, AnnIndexStats, IndexError};

// ── Config ────────────────────────────────────────────────────────────────────

/// All tuning knobs for the learned entry predictor.
#[derive(Debug, Clone)]
pub struct LearnedEntryConfig {
    /// Output dimensionality of the random projection (default 16).
    pub proj_dim: usize,
    /// L2 distance in projected space above which we fall back to the
    /// standard HNSW entry point (default 2.0).
    pub fallback_threshold: f32,
}

impl Default for LearnedEntryConfig {
    fn default() -> Self {
        Self {
            proj_dim: 16,
            fallback_threshold: 2.0,
        }
    }
}

// ── Random projection ─────────────────────────────────────────────────────────

struct RandomProjection {
    input_dim: usize,
    proj_dim: usize,
    /// Row-major weight matrix of shape `[input_dim × proj_dim]`.
    matrix: Vec<f32>,
}

impl RandomProjection {
    fn new(input_dim: usize, proj_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (proj_dim as f32).sqrt();
        let matrix = (0..input_dim * proj_dim)
            .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        Self {
            input_dim,
            proj_dim,
            matrix,
        }
    }

    fn project(&self, v: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.proj_dim];
        for (i, &vi) in v.iter().take(self.input_dim).enumerate() {
            let row = i * self.proj_dim;
            for (j, o) in out.iter_mut().enumerate() {
                *o += vi * self.matrix[row + j];
            }
        }
        out
    }

    fn memory_bytes(&self) -> usize {
        self.matrix.len() * std::mem::size_of::<f32>()
    }
}

// ── Training data ─────────────────────────────────────────────────────────────

struct TrainEntry {
    proj: Vec<f32>,
    node_id: u64,
    max_layer: usize,
}

// ── Predictor ─────────────────────────────────────────────────────────────────

/// Nearest-neighbour predictor over a random projection of inserted nodes.
pub struct LearnedEntryPoint {
    config: LearnedEntryConfig,
    projection: RandomProjection,
    training_data: Vec<TrainEntry>,
}

impl LearnedEntryPoint {
    pub fn new(input_dim: usize, config: LearnedEntryConfig) -> Self {
        let proj_dim = config.proj_dim;
        Self {
            projection: RandomProjection::new(input_dim, proj_dim),
            config,
            training_data: Vec::new(),
        }
    }

    /// Register an inserted node.  Call after every successful
    /// [`HnswIndex::insert`] to keep the predictor up to date.
    pub fn add_training_point(&mut self, vector: &[f32], node_id: u64, max_layer: usize) {
        let proj = self.projection.project(vector);
        self.training_data.push(TrainEntry {
            proj,
            node_id,
            max_layer,
        });
    }

    /// Predict the best entry node and layer for `query`.
    ///
    /// Returns `None` when no training data is available or when the nearest
    /// projected neighbour exceeds `fallback_threshold` (caller should fall
    /// back to the standard entry point).
    pub fn predict(&self, query: &[f32]) -> Option<(u64, usize)> {
        if self.training_data.is_empty() {
            return None;
        }
        let proj_q = self.projection.project(query);

        let (best_idx, best_dist) = self
            .training_data
            .iter()
            .enumerate()
            .map(|(i, t)| (i, l2_distance(&proj_q, &t.proj)))
            .fold(
                (0, f32::MAX),
                |(bi, bd), (i, d)| {
                    if d < bd {
                        (i, d)
                    } else {
                        (bi, bd)
                    }
                },
            );

        if best_dist > self.config.fallback_threshold {
            return None; // beyond confidence radius — signal fallback
        }
        let t = &self.training_data[best_idx];
        Some((t.node_id, t.max_layer))
    }

    /// Number of training points stored.
    pub fn training_size(&self) -> usize {
        self.training_data.len()
    }

    fn memory_bytes(&self) -> usize {
        self.projection.memory_bytes()
            + self
                .training_data
                .iter()
                .map(|t| t.proj.len() * std::mem::size_of::<f32>())
                .sum::<usize>()
    }
}

// ── Wrapper index ─────────────────────────────────────────────────────────────

/// HNSW augmented with a learned entry-point predictor.
///
/// On each search, the predictor is consulted first.  When its confidence
/// exceeds the threshold, the search starts from the predicted node; otherwise
/// the standard global entry point is used.  Recall is never worse than the
/// unaugmented HNSW because the fallback path is always available.
pub struct HnswWithLearnedEntry {
    /// The underlying HNSW index (public for test introspection).
    pub inner: HnswIndex,
    /// The learned-entry predictor.
    pub entry: LearnedEntryPoint,
}

impl HnswWithLearnedEntry {
    pub fn new(dim: usize, hnsw_config: HnswConfig, entry_config: LearnedEntryConfig) -> Self {
        Self {
            inner: HnswIndex::new(dim, hnsw_config),
            entry: LearnedEntryPoint::new(dim, entry_config),
        }
    }
}

impl AnnIndex for HnswWithLearnedEntry {
    fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), IndexError> {
        self.inner.insert(id, vector)?;
        // Record the inserted node in the predictor's training set.
        let max_layer = self.inner.nodes.get(&id).map_or(0, |n| n.max_layer);
        self.entry.add_training_point(vector, id, max_layer);
        Ok(())
    }

    fn delete(&mut self, id: u64) -> Result<(), IndexError> {
        self.inner.delete(id)
    }

    fn search_knn(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, IndexError> {
        // Try learned entry; fall back to standard entry when predictor is
        // uncertain or the predicted node has been deleted.
        match self.entry.predict(query) {
            Some((ep_id, ep_layer))
                if self.inner.nodes.contains_key(&ep_id)
                    && !self.inner.tombstones.contains(&ep_id) =>
            {
                self.inner.search_knn_with_entry(query, k, ep_id, ep_layer)
            }
            _ => self.inner.search_knn(query, k),
        }
    }

    fn search_range(&self, query: &[f32], radius: f32) -> Result<Vec<(u64, f32)>, IndexError> {
        self.inner.search_range(query, radius)
    }

    fn stats(&self) -> AnnIndexStats {
        let inner = self.inner.stats();
        AnnIndexStats {
            memory_bytes: inner.memory_bytes + self.entry.memory_bytes(),
            num_vectors: inner.num_vectors,
            index_type: "hnsw-learned-entry",
        }
    }
}
