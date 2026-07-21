//! In-memory HNSW (Hierarchical Navigable Small World) index.
//!
//! Implements the algorithm from Malkov & Yashunin (2018) including the
//! neighbour-diversity heuristic (Algorithm 4 in the paper).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rand::Rng;
use serde::{Deserialize, Serialize};
use strata_simd::l2_distance;

use crate::{AnnIndex, AnnIndexStats, DistId, IndexError};

// ── Config ────────────────────────────────────────────────────────────────────

/// All HNSW tuning knobs in one place.  Avoids multi-positional-arg functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Max neighbours per node at layers ≥ 1 (paper's *M*).
    pub m: usize,
    /// Max neighbours per node at layer 0 (typically `2 * m`).
    pub m0: usize,
    /// Beam width during construction (*efConstruction*).
    pub ef_construction: usize,
    /// Beam width during search (*ef*).
    pub ef_search: usize,
    /// Level-generation factor; paper recommends `1 / ln(m)`.
    pub ml_factor: f64,
}

impl HnswConfig {
    /// Build config with the canonical level factor for the given `m`.
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            m,
            m0: 2 * m,
            ef_construction,
            ef_search,
            ml_factor: 1.0 / (m as f64).ln(),
        }
    }
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self::new(16, 200, 100)
    }
}

// ── Node (crate-private fields used by the learned-entry wrapper) ─────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct HnswNode {
    pub(crate) vector: Vec<f32>,
    pub(crate) max_layer: usize,
}

// ── Index ─────────────────────────────────────────────────────────────────────

/// In-memory HNSW index with lazy tombstone-based deletion.
#[derive(Debug, Serialize, Deserialize)]
pub struct HnswIndex {
    config: HnswConfig,
    pub(crate) dim: usize,
    pub(crate) nodes: HashMap<u64, HnswNode>,
    /// `layers[lc]` maps `node_id → neighbour_ids` at layer `lc`.
    layers: Vec<HashMap<u64, Vec<u64>>>,
    pub(crate) entry_point: Option<u64>,
    pub(crate) entry_layer: usize,
    pub(crate) tombstones: HashSet<u64>,
}

impl HnswIndex {
    pub fn new(dim: usize, config: HnswConfig) -> Self {
        Self {
            config,
            dim,
            nodes: HashMap::new(),
            layers: Vec::new(),
            entry_point: None,
            entry_layer: 0,
            tombstones: HashSet::new(),
        }
    }

    /// Serialise the entire index to bytes (bincode).
    pub fn to_bytes(&self) -> Result<Vec<u8>, IndexError> {
        bincode::serialize(self).map_err(|e| IndexError::Other(e.to_string()))
    }

    /// Deserialise an index previously produced by [`to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IndexError> {
        bincode::deserialize(bytes).map_err(|e| IndexError::Other(e.to_string()))
    }

    /// Number of live (non-tombstoned) vectors.
    pub fn len(&self) -> usize {
        self.nodes.len().saturating_sub(self.tombstones.len())
    }

    /// `true` when no live vectors are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ── Layer-level beam search ───────────────────────────────────────────────

    /// Beam search within one HNSW layer.  Returns up to `ef` nearest IDs,
    /// nearest-first.  Tombstoned nodes are skipped.
    pub(crate) fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u64],
        ef: usize,
        layer: usize,
    ) -> Vec<u64> {
        // min-heap: expand nearest unvisited node first.
        let mut candidates: BinaryHeap<Reverse<DistId>> = BinaryHeap::new();
        // max-heap bounded to `ef`: worst element on top for O(1) pruning check.
        let mut result: BinaryHeap<DistId> = BinaryHeap::new();
        let mut visited: HashSet<u64> = HashSet::new();

        for &ep in entry_points {
            if self.tombstones.contains(&ep) {
                continue;
            }
            if let Some(node) = self.nodes.get(&ep) {
                let d = l2_distance(query, &node.vector);
                let did = DistId { dist: d, id: ep };
                candidates.push(Reverse(did));
                result.push(did);
                visited.insert(ep);
            }
        }

        while let Some(Reverse(c)) = candidates.pop() {
            let worst = result.peek().map_or(f32::MAX, |w| w.dist);
            if c.dist > worst {
                break; // all remaining candidates are farther than the worst result
            }

            let layer_map = match self.layers.get(layer) {
                Some(m) => m,
                None => continue,
            };
            if let Some(neighbours) = layer_map.get(&c.id) {
                for &nb_id in neighbours {
                    if visited.contains(&nb_id) || self.tombstones.contains(&nb_id) {
                        continue;
                    }
                    visited.insert(nb_id);
                    if let Some(nb_node) = self.nodes.get(&nb_id) {
                        let nb_d = l2_distance(query, &nb_node.vector);
                        let worst = result.peek().map_or(f32::MAX, |w| w.dist);
                        if nb_d < worst || result.len() < ef {
                            candidates.push(Reverse(DistId {
                                dist: nb_d,
                                id: nb_id,
                            }));
                            result.push(DistId {
                                dist: nb_d,
                                id: nb_id,
                            });
                            if result.len() > ef {
                                result.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<(f32, u64)> = result.into_iter().map(|d| (d.dist, d.id)).collect();
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        out.into_iter().map(|(_, id)| id).collect()
    }

    // ── Neighbour-diversity heuristic (paper Algorithm 4) ─────────────────────

    /// Select up to `m_max` neighbours from `candidates` using the diversity
    /// criterion: add a candidate only when it is closer to `query` than to
    /// every already-chosen neighbour.  Pruned candidates back-fill to `m_max`
    /// (`keepPrunedConnections = true`).
    fn select_neighbours_heuristic(
        &self,
        query: &[f32],
        candidates: impl Iterator<Item = u64>,
        m_max: usize,
    ) -> Vec<u64> {
        let mut sorted: Vec<(f32, u64)> = candidates
            .filter(|id| !self.tombstones.contains(id))
            .filter_map(|id| {
                self.nodes
                    .get(&id)
                    .map(|n| (l2_distance(query, &n.vector), id))
            })
            .collect();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut result: Vec<u64> = Vec::with_capacity(m_max);
        let mut discarded: Vec<u64> = Vec::new();

        for &(dist_eq, cand_id) in &sorted {
            if result.len() >= m_max {
                break;
            }
            let cand_vec = match self.nodes.get(&cand_id) {
                Some(n) => &n.vector,
                None => continue,
            };
            // Keep iff closer to query than to every already-chosen neighbour.
            let useful = result
                .iter()
                .all(|r_id| dist_eq < l2_distance(cand_vec, &self.nodes[r_id].vector));
            if useful {
                result.push(cand_id);
            } else {
                discarded.push(cand_id);
            }
        }

        // keepPrunedConnections: top-up result from the discarded set.
        for cand_id in discarded {
            if result.len() >= m_max {
                break;
            }
            result.push(cand_id);
        }
        result
    }

    // ── Search from an explicit entry point (used by learned-entry wrapper) ───

    /// Identical to `search_knn` but starts from `start_id` at `start_layer`
    /// instead of the global entry point.
    pub(crate) fn search_knn_with_entry(
        &self,
        query: &[f32],
        k: usize,
        start_id: u64,
        start_layer: usize,
    ) -> Result<Vec<(u64, f32)>, IndexError> {
        if query.len() != self.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        let top = start_layer.min(self.entry_layer);
        let mut ep_list = vec![start_id];

        for lc in (1..=top).rev() {
            let w = self.search_layer(query, &ep_list, 1, lc);
            if !w.is_empty() {
                ep_list = vec![w[0]];
            }
        }

        let ef = self.config.ef_search.max(k);
        let w = self.search_layer(query, &ep_list, ef, 0);
        let mut results = self.score_and_filter(query, &w, k);
        results.truncate(k);
        Ok(results)
    }

    fn score_and_filter(&self, query: &[f32], ids: &[u64], k: usize) -> Vec<(u64, f32)> {
        let mut v: Vec<(u64, f32)> = ids
            .iter()
            .filter(|id| !self.tombstones.contains(id))
            .filter_map(|&id| {
                self.nodes
                    .get(&id)
                    .map(|n| (id, l2_distance(query, &n.vector)))
            })
            .collect();
        v.sort_by(|a, b| a.1.total_cmp(&b.1));
        v.truncate(k);
        v
    }
}

// ── AnnIndex impl ─────────────────────────────────────────────────────────────

impl AnnIndex for HnswIndex {
    fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), IndexError> {
        if vector.len() != self.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        self.tombstones.remove(&id);

        // Random level: floor(-ln(U) * ml_factor), capped one above current top.
        let max_layer = {
            let u: f64 = rand::thread_rng().gen::<f64>().max(f64::MIN_POSITIVE);
            let level = (-u.ln() * self.config.ml_factor).floor() as usize;
            level.min(self.entry_layer + 1)
        };

        while self.layers.len() <= max_layer {
            self.layers.push(HashMap::new());
        }
        self.nodes.insert(
            id,
            HnswNode {
                vector: vector.to_vec(),
                max_layer,
            },
        );

        // ── First node ───────────────────────────────────────────────────────
        if self.entry_point.is_none() {
            for l in 0..=max_layer {
                self.layers[l].insert(id, Vec::new());
            }
            self.entry_point = Some(id);
            self.entry_layer = max_layer;
            return Ok(());
        }

        let ep = self.entry_point.unwrap(); // safe: checked is_none() above
        let entry_layer = self.entry_layer;

        // Phase 1: greedy descent from top to max_layer+1 with ef=1.
        let mut ep_list = vec![ep];
        if max_layer < entry_layer {
            for lc in (max_layer + 1..=entry_layer).rev() {
                let w = self.search_layer(vector, &ep_list, 1, lc);
                if !w.is_empty() {
                    ep_list = vec![w[0]];
                }
            }
        }

        // Phase 2: full beam search from min(entry_layer, max_layer) down to 0.
        for lc in (0..=max_layer.min(entry_layer)).rev() {
            let m_max = if lc == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let w = self.search_layer(vector, &ep_list, self.config.ef_construction, lc);
            let neighbours = self.select_neighbours_heuristic(vector, w.iter().copied(), m_max);

            self.layers[lc].insert(id, neighbours.clone());

            for &nb_id in &neighbours {
                {
                    let entry = self.layers[lc].entry(nb_id).or_default();
                    entry.push(id);
                }
                let exceeds = self.layers[lc].get(&nb_id).is_some_and(|v| v.len() > m_max);
                if exceeds {
                    let current = self.layers[lc][&nb_id].clone();
                    let nb_vec = self.nodes[&nb_id].vector.clone();
                    let pruned =
                        self.select_neighbours_heuristic(&nb_vec, current.into_iter(), m_max);
                    self.layers[lc].insert(nb_id, pruned);
                }
            }

            ep_list = w;
        }

        // Promote entry point when new node reaches a higher layer.
        if max_layer > entry_layer {
            for lc in entry_layer + 1..=max_layer {
                self.layers[lc].insert(id, Vec::new());
            }
            self.entry_point = Some(id);
            self.entry_layer = max_layer;
        }

        Ok(())
    }

    fn delete(&mut self, id: u64) -> Result<(), IndexError> {
        if !self.nodes.contains_key(&id) && !self.tombstones.contains(&id) {
            return Err(IndexError::NotFound(id));
        }
        // Lazy deletion: mark as tombstone; filtered at search and prune time.
        self.tombstones.insert(id);
        Ok(())
    }

    fn search_knn(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, IndexError> {
        if query.len() != self.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }

        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };

        let mut ep_list = vec![ep];
        for lc in (1..=self.entry_layer).rev() {
            let w = self.search_layer(query, &ep_list, 1, lc);
            if !w.is_empty() {
                ep_list = vec![w[0]];
            }
        }

        let ef = self.config.ef_search.max(k);
        let w = self.search_layer(query, &ep_list, ef, 0);
        let mut results = self.score_and_filter(query, &w, k);
        results.truncate(k);
        Ok(results)
    }

    fn search_range(&self, query: &[f32], radius: f32) -> Result<Vec<(u64, f32)>, IndexError> {
        let k = self.nodes.len().max(1);
        let all = self.search_knn(query, k)?;
        Ok(all.into_iter().filter(|(_, d)| *d <= radius).collect())
    }

    fn stats(&self) -> AnnIndexStats {
        let node_bytes: usize = self
            .nodes
            .values()
            .map(|n| n.vector.len() * std::mem::size_of::<f32>() + 16)
            .sum();
        let adj_bytes: usize = self
            .layers
            .iter()
            .flat_map(|layer| layer.values())
            .map(|v| v.len() * std::mem::size_of::<u64>())
            .sum();
        AnnIndexStats {
            memory_bytes: node_bytes + adj_bytes,
            num_vectors: self.len(),
            index_type: "hnsw",
        }
    }
}
