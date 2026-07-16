use std::fmt;

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

pub trait AnnIndex: Send + Sync {
    fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), IndexError>;
    fn delete(&mut self, id: u64) -> Result<(), IndexError>;
    fn search_knn(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, IndexError>;
    fn search_range(&self, query: &[f32], radius: f32) -> Result<Vec<(u64, f32)>, IndexError>;
}
