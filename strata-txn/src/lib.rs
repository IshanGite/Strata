use std::fmt;
pub use strata_storage::HlcTimestamp;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Mutation {
    Put(Vec<u8>, Vec<u8>),
    Delete(Vec<u8>),
}

#[derive(Debug)]
pub enum TxnError {
    LockConflict {
        key: Vec<u8>,
        primary: Vec<u8>,
        lock_ts: HlcTimestamp,
    },
    WriteConflict {
        key: Vec<u8>,
        conflict_ts: HlcTimestamp,
    },
    Aborted,
    Storage(strata_storage::StorageError),
    Other(String),
}

impl fmt::Display for TxnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TxnError::LockConflict {
                key,
                primary,
                lock_ts,
            } => {
                write!(
                    f,
                    "Lock conflict on key {:?}, primary keys {:?}, lock_ts: {}",
                    key, primary, lock_ts
                )
            }
            TxnError::WriteConflict { key, conflict_ts } => {
                write!(
                    f,
                    "Write conflict on key {:?}, conflict_ts: {}",
                    key, conflict_ts
                )
            }
            TxnError::Aborted => write!(f, "Transaction aborted"),
            TxnError::Storage(e) => write!(f, "Storage error: {}", e),
            TxnError::Other(e) => write!(f, "Txn error: {}", e),
        }
    }
}

impl std::error::Error for TxnError {}

pub trait TransactionCoordinator: Send + Sync {
    fn begin(&self) -> HlcTimestamp;
    fn prewrite(&self, txn_ts: HlcTimestamp, mutations: &[Mutation]) -> Result<(), TxnError>; // phase 1 of 2PC
    fn commit(&self, txn_ts: HlcTimestamp, commit_ts: HlcTimestamp) -> Result<(), TxnError>; // phase 2 of 2PC
    fn abort(&self, txn_ts: HlcTimestamp) -> Result<(), TxnError>;
}
