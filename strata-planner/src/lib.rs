//! strata-planner: Cost-based Query Planner for Strata Vector Database.
//!
//! Given a vector query (with optional metadata filters and range bounds),
//! chooses the optimal execution strategy based on shard cardinality and selectivity
//! statistics.

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use strata_index::{
    hnsw::HnswIndex, learned_entry::HnswWithLearnedEntry, vamana_disk::VamanaDiskIndex, AnnIndex,
    IndexError,
};
use strata_simd::l2_distance;

/// Represents a vector search query with optional range bounds and metadata filters.
#[derive(Debug, Clone)]
pub struct VectorQuery {
    pub vector: Vec<f32>,
    pub k: usize,
    pub radius: Option<f32>,
    pub filter: Option<RoaringBitmap>,
}

impl VectorQuery {
    pub fn new(vector: Vec<f32>, k: usize) -> Self {
        Self {
            vector,
            k,
            radius: None,
            filter: None,
        }
    }

    pub fn with_radius(mut self, radius: f32) -> Self {
        self.radius = Some(radius);
        self
    }

    pub fn with_filter(mut self, filter: RoaringBitmap) -> Self {
        self.filter = Some(filter);
        self
    }
}

/// Execution strategy chosen by the cost-based query planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    /// Brute-force scan (cheaper for small collections/shards below threshold).
    BruteForceScan,
    /// Standard in-memory HNSW search.
    InMemoryHnsw,
    /// Learned-entry HNSW search.
    InMemoryLearnedHnsw,
    /// Out-of-core Vamana + PQ index for memory-exceeding collections.
    OutofCoreVamana,
    /// Pre-filter graph search (bitmap filtered during graph traversal).
    PrefilterGraph,
    /// Post-filter graph search (ANN search followed by filter application).
    PostfilterGraph,
}

/// Shard cardinality and selectivity statistics.
#[derive(Debug, Clone)]
pub struct ShardStats {
    pub row_count: usize,
    pub memory_bytes: usize,
    pub has_disk_index: bool,
    pub filter_histograms: HashMap<String, f64>,
}

impl ShardStats {
    pub fn new(row_count: usize, memory_bytes: usize, has_disk_index: bool) -> Self {
        Self {
            row_count,
            memory_bytes,
            has_disk_index,
            filter_histograms: HashMap::new(),
        }
    }

    pub fn estimate_selectivity(&self, filter: &RoaringBitmap) -> f64 {
        if self.row_count == 0 {
            return 0.0;
        }
        let matching = filter.len() as f64;
        (matching / self.row_count as f64).clamp(0.0, 1.0)
    }

    pub fn refresh(&mut self, row_count: usize, memory_bytes: usize) {
        self.row_count = row_count;
        self.memory_bytes = memory_bytes;
    }
}

/// Planner configuration parameters.
#[derive(Debug, Clone)]
pub struct PlannerConfig {
    /// Threshold below which brute force scan is chosen over graph indexes.
    pub bruteforce_threshold: usize,
    /// Memory threshold (in bytes) above which out-of-core Vamana is used.
    pub memory_threshold_bytes: usize,
    /// Selectivity threshold for pre-filtering vs post-filtering.
    /// Below this selectivity, pre-filtering is used; above it, post-filtering is used.
    pub prefilter_selectivity_threshold: f64,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            bruteforce_threshold: 500,
            memory_threshold_bytes: 10 * 1024 * 1024, // 10 MB default
            prefilter_selectivity_threshold: 0.25,
        }
    }
}

/// Cost-based Query Planner.
#[derive(Debug, Clone)]
pub struct QueryPlanner {
    pub config: PlannerConfig,
}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryPlanner {
    pub fn new() -> Self {
        Self {
            config: PlannerConfig::default(),
        }
    }

    pub fn with_config(config: PlannerConfig) -> Self {
        Self { config }
    }

    /// Choose execution strategy based on query and shard stats.
    pub fn plan(&self, query: &VectorQuery, stats: &ShardStats) -> ExecutionStrategy {
        // 1. Small dataset scan check
        if stats.row_count <= self.config.bruteforce_threshold {
            return ExecutionStrategy::BruteForceScan;
        }

        // 2. Out-of-core memory threshold check
        if stats.memory_bytes > self.config.memory_threshold_bytes && stats.has_disk_index {
            return ExecutionStrategy::OutofCoreVamana;
        }

        // 3. Filtered search strategy check
        if let Some(ref filter) = query.filter {
            let selectivity = stats.estimate_selectivity(filter);
            if selectivity < self.config.prefilter_selectivity_threshold {
                return ExecutionStrategy::PrefilterGraph;
            } else {
                return ExecutionStrategy::PostfilterGraph;
            }
        }

        // 4. Default in-memory index strategy
        ExecutionStrategy::InMemoryHnsw
    }

    /// Estimate cost of a strategy (in arbitrary cost units / estimated latency).
    pub fn estimate_cost(
        &self,
        strategy: ExecutionStrategy,
        query: &VectorQuery,
        stats: &ShardStats,
    ) -> f64 {
        let n = stats.row_count as f64;
        let k = query.k as f64;

        match strategy {
            ExecutionStrategy::BruteForceScan => n * 1.0,
            ExecutionStrategy::InMemoryHnsw | ExecutionStrategy::InMemoryLearnedHnsw => {
                let ef = (k * 2.0).max(64.0);
                (n.ln() + 1.0) * ef * 1.5
            }
            ExecutionStrategy::OutofCoreVamana => {
                let ef = (k * 3.0).max(64.0);
                (n.ln() + 1.0) * ef * 4.0
            }
            ExecutionStrategy::PrefilterGraph => {
                let sel = query
                    .filter
                    .as_ref()
                    .map_or(1.0, |f| stats.estimate_selectivity(f));
                let node_check_factor = if sel < 0.05 { 8.0 } else { 2.0 / (sel + 0.1) };
                (n.ln() + 1.0) * k * 2.0 * node_check_factor
            }
            ExecutionStrategy::PostfilterGraph => {
                let sel = query
                    .filter
                    .as_ref()
                    .map_or(1.0, |f| stats.estimate_selectivity(f));
                let overfetch = (k / sel.max(0.001)).min(n);
                (n.ln() + 1.0) * overfetch * 1.5
            }
        }
    }

    /// Execute brute-force scan on a slice of vectors.
    pub fn execute_bruteforce(
        &self,
        dataset: &[(u64, Vec<f32>)],
        query: &VectorQuery,
    ) -> Vec<(u64, f32)> {
        let mut dists: Vec<(u64, f32)> = dataset
            .iter()
            .filter(|(id, _)| {
                if let Some(ref filter) = query.filter {
                    filter.contains(*id as u32)
                } else {
                    true
                }
            })
            .map(|(id, v)| (*id, l2_distance(&query.vector, v)))
            .filter(|(_, d)| {
                if let Some(r) = query.radius {
                    *d <= r
                } else {
                    true
                }
            })
            .collect();

        dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        dists.truncate(query.k);
        dists
    }

    /// Execute pre-filtered graph search on HNSW.
    pub fn execute_prefilter_hnsw(
        &self,
        hnsw: &HnswIndex,
        query: &VectorQuery,
    ) -> Result<Vec<(u64, f32)>, IndexError> {
        let raw = if let Some(ref filter) = query.filter {
            hnsw.search_knn_filtered(&query.vector, query.k, filter)?
        } else {
            hnsw.search_knn(&query.vector, query.k)?
        };

        Ok(self.apply_radius(raw, query.radius))
    }

    /// Execute post-filtered graph search on HNSW.
    pub fn execute_postfilter_hnsw(
        &self,
        hnsw: &HnswIndex,
        query: &VectorQuery,
        stats: &ShardStats,
    ) -> Result<Vec<(u64, f32)>, IndexError> {
        let sel = query
            .filter
            .as_ref()
            .map_or(1.0, |f| stats.estimate_selectivity(f));
        let overfetch_k = ((query.k as f64 / sel.max(0.01)) as usize)
            .max(query.k)
            .min(stats.row_count);

        let raw = hnsw.search_knn(&query.vector, overfetch_k)?;

        let filtered: Vec<(u64, f32)> = raw
            .into_iter()
            .filter(|(id, _)| {
                if let Some(ref filter) = query.filter {
                    filter.contains(*id as u32)
                } else {
                    true
                }
            })
            .collect();

        let mut res = self.apply_radius(filtered, query.radius);
        res.truncate(query.k);
        Ok(res)
    }

    /// Execute search on Learned Entry HNSW index.
    pub fn execute_learned_hnsw(
        &self,
        learned_hnsw: &HnswWithLearnedEntry,
        query: &VectorQuery,
    ) -> Result<Vec<(u64, f32)>, IndexError> {
        let raw = if let Some(ref filter) = query.filter {
            learned_hnsw.search_knn_filtered(&query.vector, query.k, filter)?
        } else {
            learned_hnsw.search_knn(&query.vector, query.k)?
        };

        Ok(self.apply_radius(raw, query.radius))
    }

    /// Execute search on Vamana disk index.
    pub fn execute_vamana_disk(
        &self,
        vamana: &VamanaDiskIndex,
        query: &VectorQuery,
    ) -> Result<Vec<(u64, f32)>, IndexError> {
        let raw = vamana.search_knn(&query.vector, query.k)?;

        let filtered: Vec<(u64, f32)> = raw
            .into_iter()
            .filter(|(id, _)| {
                if let Some(ref filter) = query.filter {
                    filter.contains(*id as u32)
                } else {
                    true
                }
            })
            .collect();

        Ok(self.apply_radius(filtered, query.radius))
    }

    fn apply_radius(&self, results: Vec<(u64, f32)>, radius: Option<f32>) -> Vec<(u64, f32)> {
        if let Some(r) = radius {
            results.into_iter().filter(|(_, d)| *d <= r).collect()
        } else {
            results
        }
    }
}
