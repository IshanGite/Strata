//! Out-of-core Vamana disk index (DiskANN-style).
//!
//! # Two-stage search
//!
//! 1. **Coarse pass** — beam search using in-memory PQ codes (ADC distances).
//!    No full-precision vectors are loaded; only a fixed PQ code table and the
//!    graph's adjacency records are touched.
//! 2. **Re-rank** — batch-read full-precision vectors for the top candidates,
//!    compute exact L2, return the true top-k.
//!
//! This two-stage pattern is DiskANN's core recall trick: the PQ coarse pass
//! is fast but lossy; the re-rank restores precision.
//!
//! # SSD-aware I/O batching
//!
//! At each beam-search step we pop up to `beam_width` candidates from the
//! frontier, sort their graph-file offsets ascending, then issue one `pread`
//! per record in the batch.  Sorting before reading converts scattered random
//! I/O into a (mostly) sequential burst — measurably better on NVMe than one
//! syscall per node visited.
//!
//! All disk reads use `std::os::unix::fs::FileExt::read_at` (`pread(2)`), which
//! is thread-safe and does not disturb the file's current position.
//!
//! # File formats
//!
//! ## `.vamana` (graph file)
//! ```text
//! Header (28 bytes):
//!   magic        : u32  = 0x56414D41  ("VAMA")
//!   num_points   : u64
//!   dims         : u32
//!   max_degree   : u32
//!   medoid_id    : u64  (actual node ID, not position)
//!
//! Per-node record (12 + max_degree * 8 bytes each):
//!   id             : u64
//!   neighbor_count : u32
//!   neighbor_ids   : [u64; max_degree]  (unused slots = u64::MAX)
//! ```
//!
//! ## `.vecs` (full-precision vectors)
//! ```text
//! Per-node record (8 + dims * 4 bytes each, same order as .vamana):
//!   id     : u64
//!   vector : [f32; dims]
//! ```
//!
//! ## `.pq` — bincode-serialized [`strata_simd::ProductQuantizer`]
//!
//! ## `.pqc` (PQ codes)
//! ```text
//! Header:
//!   num_points : u64
//!   m          : u32
//! Per-node record (8 + m bytes each):
//!   id    : u64
//!   codes : [u8; m]
//! ```

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::FileExt;

use strata_simd::{l2_distance, ProductQuantizer};

use crate::vamana::{build_vamana_graph, VamanaConfig, VamanaGraph};
use crate::{AnnIndex, AnnIndexStats, DistId, IndexError};

const MAGIC: u32 = 0x5641_4D41;
/// Byte size of the graph-file header.
const GRAPH_HEADER_LEN: u64 = 28; // 4 + 8 + 4 + 4 + 8

// ── Index struct ──────────────────────────────────────────────────────────────

/// Out-of-core Vamana ANN index backed by graph and vector files on disk.
///
/// During search, only PQ codes (`m × num_points` bytes) and a small offset
/// map are held in memory.  Full-precision vectors are read from disk only for
/// the final re-rank of the top candidates.
pub struct VamanaDiskIndex {
    config: VamanaConfig,
    graph_path: PathBuf,
    vecs_path: PathBuf,
    /// PQ quantizer (codebooks only, ≪ 1 MB).
    pq: ProductQuantizer,
    /// Flat PQ codes: `pq_codes[pos * m .. (pos+1) * m]` for node at `pos`.
    pq_codes: Vec<u8>,
    /// `node_pos[id]` = 0-based position of that node in both files.
    node_pos: HashMap<u64, usize>,
    /// Starting node for every search.
    medoid_id: u64,
    num_points: usize,
    dims: u32,
    max_degree: u32,
    /// Number of PQ subspaces.
    m: usize,
}

impl VamanaDiskIndex {
    // ── Build ─────────────────────────────────────────────────────────────────

    /// Build a disk index from `(id, vector)` pairs and write four files under
    /// `dir` with the given `name` prefix.
    pub fn build(
        vectors: &[(u64, Vec<f32>)],
        config: VamanaConfig,
        dir: &Path,
        name: &str,
    ) -> Result<Self, IndexError> {
        assert!(
            !vectors.is_empty(),
            "Cannot build a VamanaDiskIndex from an empty dataset"
        );
        let dim = vectors[0].1.len();
        let m = config.pq_subspaces;
        assert_eq!(
            dim % m,
            0,
            "dim ({}) must be divisible by pq_subspaces ({})",
            dim,
            m
        );

        // 1. Build in-memory graph.
        let graph = build_vamana_graph(vectors, &config);

        // 2. Train PQ on the full dataset.
        let dataset_refs: Vec<&[f32]> = vectors.iter().map(|(_, v)| v.as_slice()).collect();
        let pq = ProductQuantizer::train(&dataset_refs, m);

        // 3. Establish a stable node ordering (sorted by ID → deterministic layout).
        let mut all_ids: Vec<u64> = vectors.iter().map(|(id, _)| *id).collect();
        all_ids.sort_unstable();
        let node_pos: HashMap<u64, usize> =
            all_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let max_degree = config.r as u32;
        let n = vectors.len();

        // 4. Write files.
        let graph_path = dir.join(format!("{}.vamana", name));
        let vecs_path = dir.join(format!("{}.vecs", name));
        let pq_path = dir.join(format!("{}.pq", name));
        let pqc_path = dir.join(format!("{}.pqc", name));

        write_graph_file(&graph_path, &graph, &all_ids, max_degree)?;
        write_vecs_file(&vecs_path, &graph.vectors, &all_ids)?;

        {
            let pq_bytes = bincode::serialize(&pq).map_err(|e| IndexError::Other(e.to_string()))?;
            let mut f = File::create(&pq_path).map_err(|e| IndexError::Io(e.to_string()))?;
            // Explicitly drive the write — never drop a Result silently.
            f.write_all(&pq_bytes)
                .map_err(|e| IndexError::Io(e.to_string()))?;
        }

        let pq_codes = build_and_write_pq_codes(&pqc_path, &pq, &graph.vectors, &all_ids, m)?;

        Ok(Self {
            config,
            graph_path,
            vecs_path,
            pq,
            pq_codes,
            node_pos,
            medoid_id: graph.medoid_id,
            num_points: n,
            dims: dim as u32,
            max_degree,
            m,
        })
    }

    // ── Open ──────────────────────────────────────────────────────────────────

    /// Reload an index previously built with [`build`].
    pub fn open(config: VamanaConfig, dir: &Path, name: &str) -> Result<Self, IndexError> {
        let graph_path = dir.join(format!("{}.vamana", name));
        let vecs_path = dir.join(format!("{}.vecs", name));
        let pq_path = dir.join(format!("{}.pq", name));
        let pqc_path = dir.join(format!("{}.pqc", name));

        let (num_points, dims, max_degree, medoid_id) = read_graph_header(&graph_path)?;

        let pq: ProductQuantizer = {
            let mut buf = Vec::new();
            File::open(&pq_path)
                .and_then(|mut f| f.read_to_end(&mut buf))
                .map_err(|e| IndexError::Io(e.to_string()))?;
            bincode::deserialize(&buf).map_err(|e| IndexError::Other(e.to_string()))?
        };
        let m = pq.m;

        let (pq_codes, node_pos) = load_pq_codes(&pqc_path, m)?;

        Ok(Self {
            config,
            graph_path,
            vecs_path,
            pq,
            pq_codes,
            node_pos,
            medoid_id,
            num_points,
            dims,
            max_degree,
            m,
        })
    }

    // ── Offset helpers ────────────────────────────────────────────────────────

    fn graph_record_bytes(&self) -> u64 {
        8 + 4 + self.max_degree as u64 * 8 // id + neighbor_count + neighbor_ids
    }

    fn graph_node_offset(&self, pos: usize) -> u64 {
        GRAPH_HEADER_LEN + pos as u64 * self.graph_record_bytes()
    }

    fn vecs_record_bytes(&self) -> u64 {
        8 + self.dims as u64 * 4 // id + f32 vector
    }

    fn vecs_node_offset(&self, pos: usize) -> u64 {
        pos as u64 * self.vecs_record_bytes()
    }

    // ── Disk reads ────────────────────────────────────────────────────────────

    /// Read the neighbour ID list for the node at `pos` in the graph file.
    /// Uses `pread(2)` — one syscall, no seek, thread-safe.
    #[cfg(unix)]
    fn read_neighbours_at(&self, file: &File, pos: usize) -> io::Result<Vec<u64>> {
        let record = self.graph_record_bytes() as usize;
        let mut buf = vec![0u8; record];
        file.read_at(&mut buf, self.graph_node_offset(pos))?;

        // Layout: [id:u64][neighbor_count:u32][neighbor_ids:u64 × max_degree]
        let count = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let count = count.min(self.max_degree as usize);
        let mut nbrs = Vec::with_capacity(count);
        for i in 0..count {
            let base = 12 + i * 8;
            let nb_id = u64::from_le_bytes(buf[base..base + 8].try_into().unwrap());
            if nb_id != u64::MAX {
                nbrs.push(nb_id);
            }
        }
        Ok(nbrs)
    }

    // ── Coarse beam search (PQ ADC distances, in-memory PQ codes only) ────────

    /// Beam search using PQ approximate distances.  Collects `rerank_k`
    /// candidates without touching full-precision vectors on disk.
    ///
    /// I/O batching: at each step we pop up to `beam_width` frontier nodes,
    /// sort their graph-file offsets, and issue the reads in offset order for
    /// sequential NVMe access patterns.
    #[cfg(unix)]
    fn beam_search_coarse(
        &self,
        graph_file: &File,
        query: &[f32],
        rerank_k: usize,
    ) -> io::Result<Vec<u64>> {
        let pq_table = self.pq.distance_table(query);
        let beam_width = self.config.beam_width;

        // `frontier`: min-heap — expand nearest unvisited ADC candidate first.
        let mut frontier: BinaryHeap<Reverse<DistId>> = BinaryHeap::new();
        // `result`: max-heap bounded to `rerank_k` (worst on top for O(1) check).
        let mut result: BinaryHeap<DistId> = BinaryHeap::new();
        let mut visited: HashSet<u64> = HashSet::new();

        // Seed with the medoid.
        if let Some(&medoid_pos) = self.node_pos.get(&self.medoid_id) {
            let codes = &self.pq_codes[medoid_pos * self.m..(medoid_pos + 1) * self.m];
            let d = self.pq.adc_distance(&pq_table, codes);
            frontier.push(Reverse(DistId {
                dist: d,
                id: self.medoid_id,
            }));
            result.push(DistId {
                dist: d,
                id: self.medoid_id,
            });
            visited.insert(self.medoid_id);
        }

        loop {
            // ── BATCH: pop up to beam_width candidates ────────────────────────
            let mut batch: Vec<(u64, usize)> = Vec::with_capacity(beam_width);
            while batch.len() < beam_width {
                match frontier.pop() {
                    None => break,
                    Some(Reverse(c)) => {
                        let worst = result.peek().map_or(f32::MAX, |w| w.dist);
                        if c.dist > worst && result.len() >= rerank_k {
                            break; // remaining frontier can't improve result
                        }
                        if let Some(&pos) = self.node_pos.get(&c.id) {
                            batch.push((c.id, pos));
                        }
                    }
                }
            }

            if batch.is_empty() {
                break;
            }

            // Sort by file offset → sequential pread burst (SSD-friendly).
            batch.sort_unstable_by_key(|&(_, pos)| self.graph_node_offset(pos));

            // ── Batch-read neighbour lists from disk ──────────────────────────
            // One pread syscall per node in the batch (not per neighbour):
            // this is the I/O "batching" — a tight burst of coalesced reads
            // before any CPU processing, avoiding back-and-forth between disk
            // and CPU at single-node granularity.
            let mut all_neighbours: Vec<u64> = Vec::new();
            for &(_, pos) in &batch {
                let nbrs = self.read_neighbours_at(graph_file, pos)?;
                all_neighbours.extend(nbrs);
            }

            // ── Expand neighbours using in-memory PQ codes ────────────────────
            for nb_id in all_neighbours {
                if visited.contains(&nb_id) {
                    continue;
                }
                visited.insert(nb_id);
                if let Some(&nb_pos) = self.node_pos.get(&nb_id) {
                    let codes = &self.pq_codes[nb_pos * self.m..(nb_pos + 1) * self.m];
                    let nb_d = self.pq.adc_distance(&pq_table, codes);
                    let worst = result.peek().map_or(f32::MAX, |w| w.dist);
                    if nb_d < worst || result.len() < rerank_k {
                        frontier.push(Reverse(DistId {
                            dist: nb_d,
                            id: nb_id,
                        }));
                        result.push(DistId {
                            dist: nb_d,
                            id: nb_id,
                        });
                        if result.len() > rerank_k {
                            result.pop();
                        }
                    }
                }
            }
        }

        Ok(result.into_iter().map(|d| d.id).collect())
    }

    // ── Re-rank with exact L2 distances ──────────────────────────────────────

    /// Batch-read full-precision vectors for `candidates`, compute exact L2,
    /// and return the true top-k.
    ///
    /// Reads are sorted by vecs-file offset before issuing — same sequential
    /// I/O principle as the coarse pass.
    #[cfg(unix)]
    fn rerank(
        &self,
        vecs_file: &File,
        candidates: &[u64],
        query: &[f32],
        k: usize,
    ) -> io::Result<Vec<(u64, f32)>> {
        let mut with_pos: Vec<(u64, usize)> = candidates
            .iter()
            .filter_map(|&id| self.node_pos.get(&id).map(|&pos| (id, pos)))
            .collect();
        // Sort ascending by file offset for sequential reads.
        with_pos.sort_unstable_by_key(|(_, pos)| self.vecs_node_offset(*pos));

        let dim = self.dims as usize;
        // Pre-allocate one buffer and reuse it across all reads — avoids a
        // heap allocation per node and makes the batching intent explicit.
        let mut buf = vec![0u8; 8 + dim * 4];
        let mut scored: Vec<(u64, f32)> = Vec::with_capacity(candidates.len());

        for (id, pos) in with_pos {
            // One pread call per candidate (sorted by offset above for
            // sequential I/O). We read into buf, then interpret the f32 slice.
            vecs_file.read_at(&mut buf, self.vecs_node_offset(pos))?;
            let vec: Vec<f32> = buf[8..] // skip the stored id prefix
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            scored.push((id, l2_distance(query, &vec)));
        }

        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(k);
        Ok(scored)
    }

    // ── Memory footprint ──────────────────────────────────────────────────────

    /// Bytes held in RAM during search (PQ codes + offset map + codebook).
    pub fn in_memory_bytes(&self) -> usize {
        let codes = self.pq_codes.len();
        let offsets =
            self.node_pos.len() * (std::mem::size_of::<u64>() + std::mem::size_of::<usize>());
        let codebook: usize = self
            .pq
            .codebooks
            .iter()
            .map(|cb| {
                cb.iter()
                    .map(|v| v.len() * std::mem::size_of::<f32>())
                    .sum::<usize>()
            })
            .sum();
        codes + offsets + codebook
    }
}

// ── AnnIndex impl ─────────────────────────────────────────────────────────────

impl AnnIndex for VamanaDiskIndex {
    fn insert(&mut self, _id: u64, _vector: &[f32]) -> Result<(), IndexError> {
        // Vamana indexes are built offline. Online insert is not supported.
        Err(IndexError::Other(
            "VamanaDiskIndex is built offline; use VamanaDiskIndex::build()".into(),
        ))
    }

    fn delete(&mut self, _id: u64) -> Result<(), IndexError> {
        Err(IndexError::Other(
            "VamanaDiskIndex does not support online deletion".into(),
        ))
    }

    #[cfg(unix)]
    fn search_knn(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, IndexError> {
        if query.len() != self.dims as usize {
            return Err(IndexError::DimensionMismatch {
                expected: self.dims as usize,
                actual: query.len(),
            });
        }

        // Re-open files per search so callers hold no mutable state and
        // pread(2) calls are safe even if search_knn is called concurrently.
        // Each open() is a tiny OS overhead vs. the I/O time of the search.
        let graph_file = File::open(&self.graph_path).map_err(|e| IndexError::Io(e.to_string()))?;
        let vecs_file = File::open(&self.vecs_path).map_err(|e| IndexError::Io(e.to_string()))?;

        let rerank_k = (k * self.config.rerank_factor).max(k);

        // Stage 1: coarse beam search with PQ ADC distances (no full-vec I/O).
        let candidates = self
            .beam_search_coarse(&graph_file, query, rerank_k)
            .map_err(|e| IndexError::Io(e.to_string()))?;

        // Stage 2: exact re-rank — batch-read full-precision vectors from disk.
        self.rerank(&vecs_file, &candidates, query, k)
            .map_err(|e| IndexError::Io(e.to_string()))
    }

    #[cfg(not(unix))]
    fn search_knn(&self, _query: &[f32], _k: usize) -> Result<Vec<(u64, f32)>, IndexError> {
        Err(IndexError::Other(
            "VamanaDiskIndex requires a Unix OS (pread-based I/O)".into(),
        ))
    }

    fn search_range(&self, query: &[f32], radius: f32) -> Result<Vec<(u64, f32)>, IndexError> {
        let results = self.search_knn(query, self.num_points.max(1))?;
        Ok(results.into_iter().filter(|(_, d)| *d <= radius).collect())
    }

    fn stats(&self) -> AnnIndexStats {
        AnnIndexStats {
            memory_bytes: self.in_memory_bytes(),
            num_vectors: self.num_points,
            index_type: "vamana-disk",
        }
    }
}

// ── File I/O helpers ──────────────────────────────────────────────────────────

fn write_graph_file(
    path: &Path,
    graph: &VamanaGraph,
    all_ids: &[u64],
    max_degree: u32,
) -> Result<(), IndexError> {
    let mut f = BufWriter::new(File::create(path).map_err(|e| IndexError::Io(e.to_string()))?);

    let n = all_ids.len() as u64;
    let dims = graph.dim as u32;

    // Header: magic | num_points | dims | max_degree | medoid_id
    f.write_all(&MAGIC.to_le_bytes())
        .map_err(|e| IndexError::Io(e.to_string()))?;
    f.write_all(&n.to_le_bytes())
        .map_err(|e| IndexError::Io(e.to_string()))?;
    f.write_all(&dims.to_le_bytes())
        .map_err(|e| IndexError::Io(e.to_string()))?;
    f.write_all(&max_degree.to_le_bytes())
        .map_err(|e| IndexError::Io(e.to_string()))?;
    f.write_all(&graph.medoid_id.to_le_bytes())
        .map_err(|e| IndexError::Io(e.to_string()))?;

    let empty: Vec<u64> = Vec::new();
    for &id in all_ids {
        let nbrs = graph.adj.get(&id).unwrap_or(&empty);
        let count = nbrs.len().min(max_degree as usize) as u32;

        f.write_all(&id.to_le_bytes())
            .map_err(|e| IndexError::Io(e.to_string()))?;
        f.write_all(&count.to_le_bytes())
            .map_err(|e| IndexError::Io(e.to_string()))?;
        for i in 0..max_degree as usize {
            let nb_id = nbrs.get(i).copied().unwrap_or(u64::MAX);
            f.write_all(&nb_id.to_le_bytes())
                .map_err(|e| IndexError::Io(e.to_string()))?;
        }
    }

    f.flush().map_err(|e| IndexError::Io(e.to_string()))?;
    Ok(())
}

fn write_vecs_file(
    path: &Path,
    vectors: &HashMap<u64, Vec<f32>>,
    all_ids: &[u64],
) -> Result<(), IndexError> {
    let mut f = BufWriter::new(File::create(path).map_err(|e| IndexError::Io(e.to_string()))?);
    for &id in all_ids {
        let vec = &vectors[&id];
        f.write_all(&id.to_le_bytes())
            .map_err(|e| IndexError::Io(e.to_string()))?;
        for &val in vec {
            f.write_all(&val.to_le_bytes())
                .map_err(|e| IndexError::Io(e.to_string()))?;
        }
    }
    f.flush().map_err(|e| IndexError::Io(e.to_string()))?;
    Ok(())
}

fn build_and_write_pq_codes(
    path: &Path,
    pq: &ProductQuantizer,
    vectors: &HashMap<u64, Vec<f32>>,
    all_ids: &[u64],
    m: usize,
) -> Result<Vec<u8>, IndexError> {
    let n = all_ids.len();
    let mut pq_codes: Vec<u8> = Vec::with_capacity(n * m);

    let mut f = BufWriter::new(File::create(path).map_err(|e| IndexError::Io(e.to_string()))?);
    f.write_all(&(n as u64).to_le_bytes())
        .map_err(|e| IndexError::Io(e.to_string()))?;
    f.write_all(&(m as u32).to_le_bytes())
        .map_err(|e| IndexError::Io(e.to_string()))?;

    for &id in all_ids {
        let codes = pq.encode(&vectors[&id]);
        f.write_all(&id.to_le_bytes())
            .map_err(|e| IndexError::Io(e.to_string()))?;
        f.write_all(&codes)
            .map_err(|e| IndexError::Io(e.to_string()))?;
        pq_codes.extend_from_slice(&codes);
    }

    f.flush().map_err(|e| IndexError::Io(e.to_string()))?;
    Ok(pq_codes)
}

fn read_graph_header(path: &Path) -> Result<(usize, u32, u32, u64), IndexError> {
    let mut f = File::open(path).map_err(|e| IndexError::Io(e.to_string()))?;
    let mut hdr = [0u8; 28];
    f.read_exact(&mut hdr)
        .map_err(|e| IndexError::Io(e.to_string()))?;

    let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
    if magic != MAGIC {
        return Err(IndexError::Other(
            "Invalid Vamana graph file (bad magic)".into(),
        ));
    }
    let num_points = u64::from_le_bytes(hdr[4..12].try_into().unwrap()) as usize;
    let dims = u32::from_le_bytes(hdr[12..16].try_into().unwrap());
    let max_degree = u32::from_le_bytes(hdr[16..20].try_into().unwrap());
    let medoid_id = u64::from_le_bytes(hdr[20..28].try_into().unwrap());

    Ok((num_points, dims, max_degree, medoid_id))
}

fn load_pq_codes(path: &Path, m: usize) -> Result<(Vec<u8>, HashMap<u64, usize>), IndexError> {
    let mut f = File::open(path).map_err(|e| IndexError::Io(e.to_string()))?;
    let mut hdr = [0u8; 12];
    f.read_exact(&mut hdr)
        .map_err(|e| IndexError::Io(e.to_string()))?;

    let num_points = u64::from_le_bytes(hdr[0..8].try_into().unwrap()) as usize;
    let stored_m = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
    if stored_m != m {
        return Err(IndexError::Other(format!(
            "PQ subspace mismatch: file has {}, expected {}",
            stored_m, m
        )));
    }

    let mut pq_codes: Vec<u8> = Vec::with_capacity(num_points * m);
    let mut node_pos: HashMap<u64, usize> = HashMap::with_capacity(num_points);
    let record = 8 + m;
    let mut buf = vec![0u8; record];

    for pos in 0..num_points {
        f.read_exact(&mut buf)
            .map_err(|e| IndexError::Io(e.to_string()))?;
        let id = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        pq_codes.extend_from_slice(&buf[8..]);
        node_pos.insert(id, pos);
    }

    Ok((pq_codes, node_pos))
}
