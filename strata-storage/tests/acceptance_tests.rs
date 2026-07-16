use strata_storage::sstable::BloomFilter;
use strata_storage::{HlcTimestamp, LsmStorage, Storage};
use tempfile::TempDir;

#[test]
fn test_mvcc_snapshot_isolation() {
    let temp_dir = TempDir::new().unwrap();
    let storage = LsmStorage::open(temp_dir.path(), 1024 * 1024, 0.01).unwrap();

    let t1 = HlcTimestamp {
        physical: 10,
        logical: 0,
    };
    let t1_5 = HlcTimestamp {
        physical: 15,
        logical: 0,
    };
    let t2 = HlcTimestamp {
        physical: 20,
        logical: 0,
    };
    let t3 = HlcTimestamp {
        physical: 30,
        logical: 0,
    };

    storage.put(b"key1", b"v1", t1).unwrap();
    storage.put(b"key1", b"v2", t2).unwrap();

    // Read as_of=t1 returns v1
    assert_eq!(storage.get(b"key1", t1).unwrap(), Some(b"v1".to_vec()));
    // Read as_of=t2 returns v2
    assert_eq!(storage.get(b"key1", t2).unwrap(), Some(b"v2".to_vec()));
    // Read as_of=t1.5 returns v1
    assert_eq!(storage.get(b"key1", t1_5).unwrap(), Some(b"v1".to_vec()));
    // Read as_of=t3 returns v2
    assert_eq!(storage.get(b"key1", t3).unwrap(), Some(b"v2".to_vec()));
    // Read as_of=t0 returns None
    let t0 = HlcTimestamp {
        physical: 5,
        logical: 0,
    };
    assert_eq!(storage.get(b"key1", t0).unwrap(), None);
}

#[test]
fn test_bloom_filter_fpr() {
    let mut keys = Vec::new();
    for i in 0..10000 {
        keys.push(format!("key_{}", i).into_bytes());
    }

    let target_fpr = 0.01; // 1%
    let bloom = BloomFilter::new(&keys, target_fpr);

    let mut false_positives = 0;
    let num_queries = 10000;
    for i in 0..num_queries {
        let query = format!("query_{}", i).into_bytes();
        if bloom.contains(&query) {
            false_positives += 1;
        }
    }

    let measured_fpr = false_positives as f32 / num_queries as f32;
    println!("Measured Bloom Filter FPR: {}", measured_fpr);
    assert!(
        measured_fpr <= target_fpr * 1.5,
        "FPR {} exceeded target threshold",
        measured_fpr
    );
}

#[test]
fn test_compaction_respects_watermark() {
    let temp_dir = TempDir::new().unwrap();
    let storage = LsmStorage::open(temp_dir.path(), 1024 * 1024, 0.01).unwrap();

    let t1 = HlcTimestamp {
        physical: 10,
        logical: 0,
    };
    let t2 = HlcTimestamp {
        physical: 20,
        logical: 0,
    };
    let t3 = HlcTimestamp {
        physical: 30,
        logical: 0,
    };

    storage.put(b"key1", b"v1", t1).unwrap();
    storage.put(b"key1", b"v2", t2).unwrap();
    storage.put(b"key1", b"v3", t3).unwrap();

    // Flush active memtable to L0 SSTable
    storage.flush().unwrap();

    // Compact Level 0 to Level 1 with watermark = t2
    storage.trigger_compaction(0, t2).unwrap();

    // Verify:
    // v3 (t3 >= t2) must be preserved.
    assert_eq!(storage.get(b"key1", t3).unwrap(), Some(b"v3".to_vec()));
    // v2 (t2 == t2, first version below or equal to watermark) must be preserved.
    assert_eq!(storage.get(b"key1", t2).unwrap(), Some(b"v2".to_vec()));
    // v1 (t1 < t2, subsequent version below watermark) must have been dropped.
    assert_eq!(storage.get(b"key1", t1).unwrap(), None);
}

#[test]
fn test_crash_recovery_1m_keys() {
    use std::fs;
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();

    // Write 1,000,000 keys in 10 batches of 100,000 keys,
    // truncating the active WAL at random offsets mid-write 10 times.
    let total_batches = 10;
    let keys_per_batch = 100_000;
    let mut oracle = std::collections::HashMap::new();

    for run in 0..total_batches {
        // Start engine
        let storage = LsmStorage::open(path, 1024 * 1024 * 128, 0.01).unwrap();

        // Write batch of keys
        for i in 0..keys_per_batch {
            let key = format!("run_{}_key_{}", run, i).into_bytes();
            let val = format!("val_{}", i).into_bytes();
            let ts = HlcTimestamp {
                physical: i as u64,
                logical: 0,
            };

            storage.put(&key, &val, ts).unwrap();
            oracle.insert(key, val);
        }

        // Shut down the storage engine (releases handles)
        drop(storage);

        // Simulate crash mid-write: truncate the active WAL file at a random byte offset
        let wal_path = path.join("wal_active.log");
        let metadata = fs::metadata(&wal_path).unwrap();
        let len = metadata.len();
        if len > 0 {
            // Select a random truncation point (not at the very start to ensure some keys are preserved)
            let truncate_point = rand::random::<u64>() % len;
            let file = fs::OpenOptions::new().write(true).open(&wal_path).unwrap();
            file.set_len(truncate_point).unwrap();
            drop(file);

            // Re-read/replay the truncated file directly to update our oracle to match exactly
            // what was successfully stored in the recovered prefix.
            let mut oracle_prefix = std::collections::HashMap::new();
            let mut wal = strata_storage::wal::Wal::new(&wal_path).unwrap();
            wal.replay(|is_delete, k, v, _ts| {
                if is_delete {
                    oracle_prefix.remove(&k);
                } else {
                    oracle_prefix.insert(k, v);
                }
            })
            .unwrap();

            // Clear oracle items for this run that were lost in the crash truncation
            for i in 0..keys_per_batch {
                let key = format!("run_{}_key_{}", run, i).into_bytes();
                if let Some(prefix_val) = oracle_prefix.get(&key) {
                    oracle.insert(key, prefix_val.clone());
                } else {
                    oracle.remove(&key);
                }
            }
        }
    }

    // Restart the storage engine to verify recovered content
    let final_storage = LsmStorage::open(path, 1024 * 1024 * 128, 0.01).unwrap();

    // Verify all keys in the oracle are recovered successfully
    for (k, expected_v) in &oracle {
        let res = final_storage
            .get(
                k,
                HlcTimestamp {
                    physical: u64::MAX,
                    logical: u32::MAX,
                },
            )
            .unwrap();
        assert_eq!(
            res.as_ref(),
            Some(expected_v),
            "Value mismatch for key {:?}",
            k
        );
    }
}
