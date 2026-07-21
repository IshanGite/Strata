use std::fmt;

// ── Shared ordered-float helper for BinaryHeap ────────────────────────────────
// Distances in this codebase are always finite and non-NaN, so total_cmp is safe.

#[derive(Clone, Copy, PartialEq)]
pub(crate) struct DistId {
    pub(crate) dist: f32,
    pub(crate) id: u64,
}

impl Eq for DistId {}

impl PartialOrd for DistId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.id.cmp(&other.id))
    }
}

// ── Public modules ────────────────────────────────────────────────────────────

pub mod hnsw;
pub mod learned_entry;
pub mod vamana;
pub mod vamana_disk;

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum IndexError {
    DimensionMismatch { expected: usize, actual: usize },
    NotFound(u64),
    Io(String),
    Other(String),
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            IndexError::NotFound(id) => write!(f, "ID {} not found in index", id),
            IndexError::Io(e) => write!(f, "IO error: {}", e),
            IndexError::Other(e) => write!(f, "Index error: {}", e),
        }
    }
}

impl std::error::Error for IndexError {}

// ── Runtime statistics (consumed by the query planner in Phase 7) ─────────────

#[derive(Debug, Clone)]
pub struct AnnIndexStats {
    /// Approximate in-memory footprint in bytes.
    pub memory_bytes: usize,
    /// Number of live (non-deleted) vectors.
    pub num_vectors: usize,
    /// Short tag identifying the index type (e.g. `"hnsw"`, `"vamana-disk"`).
    pub index_type: &'static str,
}

// ── Core trait ────────────────────────────────────────────────────────────────

pub trait AnnIndex: Send + Sync {
    fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), IndexError>;
    fn delete(&mut self, id: u64) -> Result<(), IndexError>;
    fn search_knn(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, IndexError>;
    fn search_range(&self, query: &[f32], radius: f32) -> Result<Vec<(u64, f32)>, IndexError>;
    fn stats(&self) -> AnnIndexStats;
}
