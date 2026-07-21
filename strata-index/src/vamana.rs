//! Vamana graph construction (DiskANN-style).
//!
//! Implements the core algorithm from:
//!   "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single
//!   Node" (Jayaram et al., NeurIPS 2019).
//!
//! The output [`VamanaGraph`] is an in-memory structure consumed by
//! [`crate::vamana_disk::VamanaDiskIndex::build`], which serialises it to disk.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rand::seq::SliceRandom;
use strata_simd::l2_distance;

use crate::DistId;

// ── Config ────────────────────────────────────────────────────────────────────

/// All Vamana tuning knobs in one place.
#[derive(Debug, Clone)]
pub struct VamanaConfig {
    /// Maximum out-degree of every node in the graph (*R* in the paper).
    pub r: usize,
    /// Pruning parameter — larger values prune more aggressively.
    /// Must be ≥ 1.0.  The paper recommends 1.2 for the first pass.
    pub alpha: f32,
    /// Search-list size during graph construction (*L* in the paper, ≥ `r`).
    pub l_build: usize,
    /// Number of PQ subspaces for in-memory coarse search (must divide `dim`).
    pub pq_subspaces: usize,
    /// Beam width for the disk beam search during queries.
    pub beam_width: usize,
    /// Oversampling factor: fetch `rerank_factor * k` coarse candidates and
    /// re-rank with exact distances.
    pub rerank_factor: usize,
}

impl Default for VamanaConfig {
    fn default() -> Self {
        Self {
            r: 32,
            alpha: 1.2,
            l_build: 75,
            pq_subspaces: 8,
            beam_width: 4,
            rerank_factor: 3,
        }
    }
}

// ── In-memory graph ───────────────────────────────────────────────────────────

/// In-memory Vamana graph produced by [`build_vamana_graph`].
pub struct VamanaGraph {
    /// `adj[node_id]` = list of neighbour IDs (up to `r`).
    pub adj: HashMap<u64, Vec<u64>>,
    /// Full-precision vectors keyed by node ID.
    pub vectors: HashMap<u64, Vec<f32>>,
    /// Medoid node ID (used as the search starting point).
    pub medoid_id: u64,
    /// Vector dimension.
    pub dim: usize,
}

// ── Build ─────────────────────────────────────────────────────────────────────

/// Build a Vamana proximity graph from `(id, vector)` pairs.
///
/// The returned [`VamanaGraph`] is suitable for serialisation via
/// [`crate::vamana_disk::VamanaDiskIndex::build`].
pub fn build_vamana_graph(vectors: &[(u64, Vec<f32>)], config: &VamanaConfig) -> VamanaGraph {
    assert!(
        !vectors.is_empty(),
        "Cannot build a Vamana graph from an empty dataset"
    );

    let dim = vectors[0].1.len();
    let n = vectors.len();

    // Borrow-friendly lookup: id → &vector.
    let id_to_vec: HashMap<u64, &Vec<f32>> = vectors.iter().map(|(id, v)| (*id, v)).collect();

    let medoid_id = find_medoid(vectors);
    let all_ids: Vec<u64> = vectors.iter().map(|(id, _)| *id).collect();

    // Initialise: give each node r random (distinct) neighbours.
    let mut adj: HashMap<u64, Vec<u64>> = HashMap::with_capacity(n);
    {
        let mut rng = rand::thread_rng();
        for &(id, _) in vectors {
            let mut pool = all_ids.clone();
            pool.retain(|&x| x != id);
            pool.shuffle(&mut rng);
            pool.truncate(config.r);
            adj.insert(id, pool);
        }
    }

    // Two-pass Vamana as described in the DiskANN paper:
    // Pass 1: α = 1.0 (less aggressive pruning → better graph connectivity)
    // Pass 2: α = config.alpha (full diversity-based pruning for quality)
    for pass in 0..2u8 {
        let alpha = if pass == 0 { 1.0f32 } else { config.alpha };
        let mut order = all_ids.clone();
        order.shuffle(&mut rand::thread_rng());

        for p_id in order {
            let p_vec = id_to_vec[&p_id];

            // Greedy search from medoid → candidate set.
            let candidates =
                greedy_search_graph(&adj, &id_to_vec, medoid_id, p_vec, config.l_build);

            // Merge candidates with current neighbours (excluding self).
            let mut v_set: HashSet<u64> = candidates.into_iter().collect();
            v_set.extend(adj[&p_id].iter().copied());
            v_set.remove(&p_id);

            // RobustPrune using this pass's alpha value.
            let new_nbrs = robust_prune(p_vec, &v_set, &id_to_vec, alpha, config.r);

            // Bidirectional edge update.
            for &q_id in &new_nbrs {
                {
                    let q_nbrs = adj.entry(q_id).or_default();
                    if !q_nbrs.contains(&p_id) {
                        q_nbrs.push(p_id);
                    }
                }
                // Prune q's list if it now exceeds r.
                if adj[&q_id].len() > config.r {
                    let q_vec = id_to_vec[&q_id];
                    let q_set: HashSet<u64> = adj[&q_id].iter().copied().collect();
                    let pruned = robust_prune(q_vec, &q_set, &id_to_vec, alpha, config.r);
                    adj.insert(q_id, pruned);
                }
            }

            adj.insert(p_id, new_nbrs);
        }
    } // end two-pass loop

    VamanaGraph {
        adj,
        vectors: vectors.iter().map(|(id, v)| (*id, v.clone())).collect(),
        medoid_id,
        dim,
    }
}

// ── RobustPrune (paper Algorithm 3) ──────────────────────────────────────────

/// Select at most `r` neighbours for node `p` from candidate set `v`.
///
/// Maintains diversity: after choosing `p*` (nearest remaining), removes any
/// candidate `v` for which `alpha * dist(p*, v) ≤ dist(p, v)` — i.e. `p*`
/// already "covers" `v` as a shortcut.
pub fn robust_prune(
    p: &[f32],
    v: &HashSet<u64>,
    id_to_vec: &HashMap<u64, &Vec<f32>>,
    alpha: f32,
    r: usize,
) -> Vec<u64> {
    let mut remaining: Vec<(f32, u64)> = v
        .iter()
        .filter_map(|&id| id_to_vec.get(&id).map(|vec| (l2_distance(p, vec), id)))
        .collect();
    remaining.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut result: Vec<u64> = Vec::with_capacity(r);

    while !remaining.is_empty() && result.len() < r {
        let (_, p_star) = remaining.remove(0); // nearest remaining candidate
        result.push(p_star);
        let p_star_vec = id_to_vec[&p_star];

        // Remove candidates dominated by p_star (retain only those NOT covered).
        remaining.retain(|&(dist_p_v, v_id)| {
            let dist_pstar_v = l2_distance(p_star_vec, id_to_vec[&v_id]);
            alpha * dist_pstar_v > dist_p_v
        });
    }

    result
}

// ── Greedy search on the current (partial) graph ──────────────────────────────

/// Beam search on the current adjacency graph, returning up to `l` nearest IDs.
pub(crate) fn greedy_search_graph(
    adj: &HashMap<u64, Vec<u64>>,
    id_to_vec: &HashMap<u64, &Vec<f32>>,
    start: u64,
    query: &[f32],
    l: usize,
) -> Vec<u64> {
    let mut visited: HashSet<u64> = HashSet::new();
    let mut candidates: BinaryHeap<Reverse<DistId>> = BinaryHeap::new();
    let mut result: BinaryHeap<DistId> = BinaryHeap::new();

    if let Some(start_vec) = id_to_vec.get(&start) {
        let d = l2_distance(query, start_vec);
        candidates.push(Reverse(DistId { dist: d, id: start }));
        result.push(DistId { dist: d, id: start });
        visited.insert(start);
    }

    while let Some(Reverse(c)) = candidates.pop() {
        let worst = result.peek().map_or(f32::MAX, |w| w.dist);
        if c.dist > worst {
            break;
        }
        if let Some(nbrs) = adj.get(&c.id) {
            for &nb_id in nbrs {
                if visited.contains(&nb_id) {
                    continue;
                }
                visited.insert(nb_id);
                if let Some(nb_vec) = id_to_vec.get(&nb_id) {
                    let nb_d = l2_distance(query, nb_vec);
                    let worst = result.peek().map_or(f32::MAX, |w| w.dist);
                    if nb_d < worst || result.len() < l {
                        candidates.push(Reverse(DistId {
                            dist: nb_d,
                            id: nb_id,
                        }));
                        result.push(DistId {
                            dist: nb_d,
                            id: nb_id,
                        });
                        if result.len() > l {
                            result.pop();
                        }
                    }
                }
            }
        }
    }

    result.into_iter().map(|d| d.id).collect()
}

// ── Medoid ────────────────────────────────────────────────────────────────────

fn find_medoid(vectors: &[(u64, Vec<f32>)]) -> u64 {
    let dim = vectors[0].1.len();
    let n = vectors.len() as f32;
    let mut centroid = vec![0.0f32; dim];
    for (_, v) in vectors {
        for (c, &x) in centroid.iter_mut().zip(v.iter()) {
            *c += x;
        }
    }
    for c in centroid.iter_mut() {
        *c /= n;
    }
    vectors
        .iter()
        .map(|(id, v)| (*id, l2_distance(v, &centroid)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(id, _)| id)
        .expect("non-empty dataset")
}
