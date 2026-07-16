pub mod compaction;
pub mod memtable;
pub mod sstable;
pub mod wal;

use compaction::compact_files;
use memtable::MemTable;
use sstable::{SsTableReader, SsTableWriter};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use wal::Wal;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct HlcTimestamp {
    pub physical: u64,
    pub logical: u32,
}

impl fmt::Display for HlcTimestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.physical, self.logical)
    }
}

#[derive(Debug)]
pub enum StorageError {
    Io(String),
    Serialization(String),
    VersionConflict,
    InvalidKey,
    Other(String),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::Io(e) => write!(f, "IO error: {}", e),
            StorageError::Serialization(e) => write!(f, "Serialization error: {}", e),
            StorageError::VersionConflict => write!(f, "Version conflict"),
            StorageError::InvalidKey => write!(f, "Invalid key"),
            StorageError::Other(e) => write!(f, "Storage error: {}", e),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<io::Error> for StorageError {
    fn from(err: io::Error) -> Self {
        StorageError::Io(err.to_string())
    }
}

pub type StorageIterator = Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>>;

pub trait Storage: Send + Sync {
    fn put(&self, key: &[u8], value: &[u8], ts: HlcTimestamp) -> Result<(), StorageError>;
    fn get(&self, key: &[u8], as_of: HlcTimestamp) -> Result<Option<Vec<u8>>, StorageError>;
    fn delete(&self, key: &[u8], ts: HlcTimestamp) -> Result<(), StorageError>;
    fn scan(
        &self,
        start: &[u8],
        end: &[u8],
        as_of: HlcTimestamp,
    ) -> Result<StorageIterator, StorageError>;
    fn gc_versions_older_than(&self, ts: HlcTimestamp) -> Result<(), StorageError>;
    fn flush(&self) -> Result<(), StorageError>;
}

struct LsmState {
    active_memtable: Arc<MemTable>,
    imm_memtables: Vec<(Arc<MemTable>, PathBuf)>, // (memtable, wal_path)
    levels: Vec<Vec<PathBuf>>,
}

pub struct LsmStorage {
    dir: PathBuf,
    state: Arc<parking_lot::RwLock<LsmState>>,
    wal: parking_lot::Mutex<Wal>,
    next_file_id: AtomicU64,
    watermark: parking_lot::Mutex<HlcTimestamp>,
    fpr: f32,
    max_memtable_size: usize,
    readers: parking_lot::Mutex<std::collections::HashMap<PathBuf, SsTableReader>>,
}

impl LsmStorage {
    pub fn open(dir: &Path, max_memtable_size: usize, fpr: f32) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let manifest_path = dir.join("MANIFEST");
        let (next_id, mut levels) = if manifest_path.exists() {
            Self::read_manifest(&manifest_path)?
        } else {
            (1, vec![Vec::new(); 7])
        };

        // Recover any immutable/dangling WALs: wal_imm_*.log
        let mut next_file_id = next_id;
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                    if filename.starts_with("wal_imm_") && filename.ends_with(".log") {
                        let mut wal = Wal::new(&path)?;
                        let mem = Arc::new(MemTable::new());
                        wal.replay(|is_delete, k, v, ts| {
                            if is_delete {
                                mem.delete(k, ts);
                            } else {
                                mem.put(k, v, ts);
                            }
                        })?;
                        // Flush it to L0
                        let sst_path = dir.join(format!("{}.sst", next_file_id));
                        let mut writer = SsTableWriter::new(&sst_path)?;
                        for (key, val) in mem.iter_all() {
                            writer.add(key, val.as_deref())?;
                        }
                        writer.finish(fpr)?;
                        levels[0].insert(0, sst_path);
                        next_file_id += 1;
                        fs::remove_file(&path)?;
                    }
                }
            }
        }

        // Recover active WAL: wal_active.log
        let active_wal_path = dir.join("wal_active.log");
        let active_memtable = Arc::new(MemTable::new());
        let mut wal = Wal::new(&active_wal_path)?;
        let mut replayed = false;
        wal.replay(|is_delete, k, v, ts| {
            replayed = true;
            if is_delete {
                active_memtable.delete(k, ts);
            } else {
                active_memtable.put(k, v, ts);
            }
        })?;

        if replayed {
            let sst_path = dir.join(format!("{}.sst", next_file_id));
            let mut writer = SsTableWriter::new(&sst_path)?;
            for (key, val) in active_memtable.iter_all() {
                writer.add(key, val.as_deref())?;
            }
            writer.finish(fpr)?;
            levels[0].insert(0, sst_path);
            next_file_id += 1;
            fs::remove_file(&active_wal_path)?;
            wal = Wal::new(&active_wal_path)?;
            active_memtable.clear();
        }

        // Update manifest if we flushed any WALs
        if next_file_id > next_id {
            Self::write_manifest(&manifest_path, next_file_id, &levels)?;
        }

        let state = Arc::new(parking_lot::RwLock::new(LsmState {
            active_memtable,
            imm_memtables: Vec::new(),
            levels,
        }));

        Ok(Self {
            dir: dir.to_path_buf(),
            state,
            wal: parking_lot::Mutex::new(wal),
            next_file_id: AtomicU64::new(next_file_id),
            watermark: parking_lot::Mutex::new(HlcTimestamp {
                physical: 0,
                logical: 0,
            }),
            fpr,
            max_memtable_size,
            readers: parking_lot::Mutex::new(std::collections::HashMap::new()),
        })
    }

    fn write_manifest(path: &Path, next_file_id: u64, levels: &[Vec<PathBuf>]) -> io::Result<()> {
        let mut f = File::create(path)?;
        writeln!(f, "{}", next_file_id)?;
        for level in levels {
            let paths: Vec<String> = level
                .iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect();
            writeln!(f, "{}", paths.join(","))?;
        }
        Ok(())
    }

    fn read_manifest(path: &Path) -> io::Result<(u64, Vec<Vec<PathBuf>>)> {
        let mut content = String::new();
        File::open(path)?.read_to_string(&mut content)?;
        let mut lines = content.lines();
        let next_file_id = lines
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Empty manifest"))?
            .parse::<u64>()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut levels = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                levels.push(Vec::new());
            } else {
                let files = line
                    .split(',')
                    .map(|s| PathBuf::from(s.trim()))
                    .filter(|p| !p.as_os_str().is_empty())
                    .collect();
                levels.push(files);
            }
        }
        while levels.len() < 7 {
            levels.push(Vec::new());
        }
        Ok((next_file_id, levels))
    }

    pub fn trigger_compaction(&self, level: usize, watermark: HlcTimestamp) -> io::Result<()> {
        if level >= 6 {
            return Ok(());
        }

        let (inputs, overlapping, is_last) = {
            let state = self.state.read();
            if state.levels[level].is_empty() {
                return Ok(());
            }
            let inputs = state.levels[level].clone();
            // Get overlapping files in level + 1
            // Since L0 can overlap wide, we get L0's overall min/max and find matches in Level 1.
            let mut min_user_key: Option<Vec<u8>> = None;
            let mut max_user_key: Option<Vec<u8>> = None;

            for path in &inputs {
                if let Ok(reader) = SsTableReader::open(path) {
                    if min_user_key.is_none()
                        || reader.min_key.user_key < min_user_key.as_ref().unwrap().clone()
                    {
                        min_user_key = Some(reader.min_key.user_key.clone());
                    }
                    if max_user_key.is_none()
                        || reader.max_key.user_key > max_user_key.as_ref().unwrap().clone()
                    {
                        max_user_key = Some(reader.max_key.user_key.clone());
                    }
                }
            }

            let mut overlapping = Vec::new();
            if let (Some(min_k), Some(max_k)) = (min_user_key, max_user_key) {
                for path in &state.levels[level + 1] {
                    if let Ok(reader) = SsTableReader::open(path) {
                        let overlap =
                            !(reader.max_key.user_key < min_k || reader.min_key.user_key > max_k);
                        if overlap {
                            overlapping.push(path.clone());
                        }
                    }
                }
            }

            (inputs, overlapping, level + 1 == 6)
        };

        let mut to_compact = inputs.clone();
        to_compact.extend(overlapping.clone());

        if to_compact.is_empty() {
            return Ok(());
        }

        let next_id_fn = || self.next_file_id.fetch_add(1, Ordering::SeqCst);
        let output_files = compact_files(
            &to_compact,
            &self.dir,
            watermark,
            is_last,
            self.fpr,
            2 * 1024 * 1024,
            next_id_fn,
        )?;

        // Update state
        {
            let mut state = self.state.write();
            // Remove compacted files
            state.levels[level].retain(|p| !inputs.contains(p));
            state.levels[level + 1].retain(|p| !overlapping.contains(p));

            // Add new files
            state.levels[level + 1].extend(output_files);
            // Sort Level 1+ by min key range to ensure non-overlapping order
            if level + 1 > 0 {
                state.levels[level + 1].sort_by(|a, b| {
                    let r_a = SsTableReader::open(a).unwrap();
                    let r_b = SsTableReader::open(b).unwrap();
                    r_a.min_key.user_key.cmp(&r_b.min_key.user_key)
                });
            }

            // Save manifest
            let next_id = self.next_file_id.load(Ordering::SeqCst);
            Self::write_manifest(&self.dir.join("MANIFEST"), next_id, &state.levels)?;
        }

        // Remove compacted input files from disk
        for path in to_compact {
            let _ = fs::remove_file(path);
        }

        Ok(())
    }
}

impl Storage for LsmStorage {
    fn put(&self, key: &[u8], value: &[u8], ts: HlcTimestamp) -> Result<(), StorageError> {
        self.wal.lock().append(false, key, value, ts)?;
        let mem = {
            let state = self.state.read();
            state.active_memtable.clone()
        };
        mem.put(key.to_vec(), value.to_vec(), ts);

        if mem.size() >= self.max_memtable_size {
            self.flush()?;
        }
        Ok(())
    }

    fn delete(&self, key: &[u8], ts: HlcTimestamp) -> Result<(), StorageError> {
        self.wal.lock().append(true, key, &[], ts)?;
        let mem = {
            let state = self.state.read();
            state.active_memtable.clone()
        };
        mem.delete(key.to_vec(), ts);
        Ok(())
    }

    fn get(&self, key: &[u8], as_of: HlcTimestamp) -> Result<Option<Vec<u8>>, StorageError> {
        let (active, imm, levels) = {
            let state = self.state.read();
            (
                state.active_memtable.clone(),
                state.imm_memtables.clone(),
                state.levels.clone(),
            )
        };

        // 1. Search active memtable
        if let Some(res) = active.get(key, as_of) {
            return Ok(res);
        }

        // 2. Search immutable memtables
        for (mem, _) in imm.iter().rev() {
            if let Some(res) = mem.get(key, as_of) {
                return Ok(res);
            }
        }

        // 3. Search Level 0 SSTables (overlapping)
        for sst_path in &levels[0] {
            let mut guard = self.readers.lock();
            let reader = if let Some(r) = guard.get_mut(sst_path) {
                r
            } else {
                let r = SsTableReader::open(sst_path)?;
                guard.insert(sst_path.clone(), r);
                guard.get_mut(sst_path).unwrap()
            };
            if let Some(res) = reader.get(key, as_of)? {
                return Ok(res);
            }
        }

        // 4. Search Level 1+
        for level in &levels[1..] {
            for sst_path in level {
                let mut guard = self.readers.lock();
                let reader = if let Some(r) = guard.get_mut(sst_path) {
                    r
                } else {
                    let r = SsTableReader::open(sst_path)?;
                    guard.insert(sst_path.clone(), r);
                    guard.get_mut(sst_path).unwrap()
                };
                if key >= reader.min_key.user_key.as_slice()
                    && key <= reader.max_key.user_key.as_slice()
                {
                    if let Some(res) = reader.get(key, as_of)? {
                        return Ok(res);
                    }
                }
            }
        }

        Ok(None)
    }

    fn scan(
        &self,
        start: &[u8],
        end: &[u8],
        as_of: HlcTimestamp,
    ) -> Result<StorageIterator, StorageError> {
        let (active, imm, levels) = {
            let state = self.state.read();
            (
                state.active_memtable.clone(),
                state.imm_memtables.clone(),
                state.levels.clone(),
            )
        };

        let mut runs = Vec::new();

        // 1. Scan active memtable
        runs.push(active.scan(start, end, as_of));

        // 2. Scan immutable memtables
        for (mem, _) in imm.iter().rev() {
            runs.push(mem.scan(start, end, as_of));
        }

        // 3. Scan L0 files
        for sst_path in &levels[0] {
            let reader = SsTableReader::open(sst_path)?;
            runs.push(scan_sstable(reader, start, end, as_of)?);
        }

        // 4. Scan Level 1+ candidate files
        for level in &levels[1..] {
            for sst_path in level {
                let reader = SsTableReader::open(sst_path)?;
                if (start.is_empty() || reader.max_key.user_key.as_slice() >= start)
                    && (end.is_empty() || reader.min_key.user_key.as_slice() < end)
                {
                    runs.push(scan_sstable(reader, start, end, as_of)?);
                }
            }
        }

        // Merge-sort the runs, preserving newest version ties and filtering tombstones
        let mut merged = Vec::new();
        let mut ptrs = vec![0; runs.len()];
        let mut last_key: Option<Vec<u8>> = None;

        loop {
            let mut min_idx: Option<usize> = None;
            for i in 0..runs.len() {
                if ptrs[i] < runs[i].len() {
                    if let Some(m_idx) = min_idx {
                        let key_i = &runs[i][ptrs[i]].0;
                        let key_min = &runs[m_idx][ptrs[m_idx]].0;
                        if key_i < key_min || (key_i == key_min && i < m_idx) {
                            min_idx = Some(i);
                        }
                    } else {
                        min_idx = Some(i);
                    }
                }
            }

            let idx = match min_idx {
                Some(i) => i,
                None => break,
            };

            let (key, value) = runs[idx][ptrs[idx]].clone();
            ptrs[idx] += 1;

            if let Some(ref last) = last_key {
                if last == &key {
                    continue;
                }
            }

            last_key = Some(key.clone());
            if let Some(val) = value {
                merged.push((key, val));
            }
        }

        Ok(Box::new(merged.into_iter()))
    }

    fn gc_versions_older_than(&self, ts: HlcTimestamp) -> Result<(), StorageError> {
        *self.watermark.lock() = ts;
        // Trigger compaction for L0 and L1 to GC older versions
        let _ = self.trigger_compaction(0, ts);
        let _ = self.trigger_compaction(1, ts);
        Ok(())
    }

    fn flush(&self) -> Result<(), StorageError> {
        let (old_memtable, old_wal_path, new_id) = {
            let mut state = self.state.write();
            let old_memtable = state.active_memtable.clone();
            let old_wal_path = self.dir.join("wal_active.log");

            // Allocate new file ID for frozen log rename
            let new_id = self.next_file_id.fetch_add(1, Ordering::SeqCst);
            let frozen_wal_path = self.dir.join(format!("wal_imm_{}.log", new_id));

            // Rename active WAL to immutable WAL on disk
            fs::rename(&old_wal_path, &frozen_wal_path)?;

            // Create new empty active memtable and active WAL
            let new_memtable = Arc::new(MemTable::new());
            let new_wal = Wal::new(&old_wal_path)?;

            // Replace in state
            state.active_memtable = new_memtable;
            state
                .imm_memtables
                .push((old_memtable.clone(), frozen_wal_path.clone()));

            let next_id_val = self.next_file_id.load(Ordering::SeqCst);
            Self::write_manifest(&self.dir.join("MANIFEST"), next_id_val, &state.levels)?;

            // Replace active WAL handle
            *self.wal.lock() = new_wal;

            (old_memtable, frozen_wal_path, new_id)
        };

        // Write frozen memtable to new SSTable
        let sst_path = self.dir.join(format!("{}.sst", new_id));
        let mut writer = SsTableWriter::new(&sst_path)?;
        for (key, val) in old_memtable.iter_all() {
            writer.add(key, val.as_deref())?;
        }
        writer.finish(self.fpr)?;

        // Update state levels and clean up frozen WAL
        {
            let mut state = self.state.write();
            state.levels[0].insert(0, sst_path);
            state.imm_memtables.retain(|(_, p)| p != &old_wal_path);

            let next_id_val = self.next_file_id.load(Ordering::SeqCst);
            Self::write_manifest(&self.dir.join("MANIFEST"), next_id_val, &state.levels)?;
        }

        let _ = fs::remove_file(old_wal_path);
        Ok(())
    }
}

#[allow(clippy::type_complexity)]
pub fn scan_sstable(
    mut reader: SsTableReader,
    start: &[u8],
    end: &[u8],
    as_of: HlcTimestamp,
) -> io::Result<Vec<(Vec<u8>, Option<Vec<u8>>)>> {
    let mut results = Vec::new();
    let mut last_seen_user_key: Option<Vec<u8>> = None;

    for i in 0..reader.index.len() {
        let entry = &reader.index[i];
        let offset = entry.offset;
        let size = entry.size;
        let entries = reader.read_block_entries(offset, size)?;
        for (key, value) in entries {
            if !start.is_empty() && key.user_key.as_slice() < start {
                continue;
            }
            if !end.is_empty() && key.user_key.as_slice() >= end {
                break;
            }

            if let Some(ref last) = last_seen_user_key {
                if last == &key.user_key {
                    continue;
                }
            }

            if key.ts <= as_of {
                results.push((key.user_key.clone(), value));
                last_seen_user_key = Some(key.user_key.clone());
            }
        }
    }
    Ok(results)
}
