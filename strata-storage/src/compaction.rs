use crate::memtable::MemKey;
use crate::sstable::{SsTableReader, SsTableWriter};
use crate::HlcTimestamp;
use std::io;
use std::path::{Path, PathBuf};

pub struct SsTableIterator {
    reader: SsTableReader,
    block_idx: usize,
    entries: Vec<(MemKey, Option<Vec<u8>>)>,
    entry_idx: usize,
}

impl SsTableIterator {
    pub fn new(reader: SsTableReader) -> io::Result<Self> {
        let mut iter = Self {
            reader,
            block_idx: 0,
            entries: Vec::new(),
            entry_idx: 0,
        };
        iter.load_next_block()?;
        Ok(iter)
    }

    fn load_next_block(&mut self) -> io::Result<()> {
        if self.block_idx >= self.reader.index.len() {
            self.entries.clear();
            self.entry_idx = 0;
            return Ok(());
        }
        let entry = &self.reader.index[self.block_idx];
        let offset = entry.offset;
        let size = entry.size;
        self.entries = self.reader.read_block_entries(offset, size)?;
        self.entry_idx = 0;
        self.block_idx += 1;
        Ok(())
    }
}

impl Iterator for SsTableIterator {
    type Item = io::Result<(MemKey, Option<Vec<u8>>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.entry_idx >= self.entries.len() {
            if let Err(e) = self.load_next_block() {
                return Some(Err(e));
            }
            if self.entries.is_empty() {
                return None;
            }
        }
        let item = self.entries[self.entry_idx].clone();
        self.entry_idx += 1;
        Some(Ok(item))
    }
}

pub struct MergeIterator {
    iters: Vec<SsTableIterator>,
    current: Vec<Option<(MemKey, Option<Vec<u8>>)>>,
}

impl MergeIterator {
    pub fn new(mut iters: Vec<SsTableIterator>) -> io::Result<Self> {
        let mut current = Vec::with_capacity(iters.len());
        for iter in &mut iters {
            if let Some(res) = iter.next() {
                current.push(Some(res?));
            } else {
                current.push(None);
            }
        }
        Ok(Self { iters, current })
    }

    pub fn next_entry(&mut self) -> io::Result<Option<(MemKey, Option<Vec<u8>>)>> {
        let mut min_idx: Option<usize> = None;
        for i in 0..self.current.len() {
            if let Some(ref val) = self.current[i] {
                if let Some(m_idx) = min_idx {
                    if let Some(ref min_val) = self.current[m_idx] {
                        if val.0 < min_val.0 {
                            min_idx = Some(i);
                        }
                    }
                } else {
                    min_idx = Some(i);
                }
            }
        }

        if let Some(idx) = min_idx {
            let result = self.current[idx].take().unwrap();
            if let Some(res) = self.iters[idx].next() {
                self.current[idx] = Some(res?);
            } else {
                self.current[idx] = None;
            }
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }
}

pub fn compact_files(
    input_paths: &[PathBuf],
    output_dir: &Path,
    watermark: HlcTimestamp,
    is_last_level: bool,
    fpr: f32,
    max_file_size: usize,
    mut next_file_id: impl FnMut() -> u64,
) -> io::Result<Vec<PathBuf>> {
    let mut iters = Vec::new();
    for path in input_paths {
        let reader = SsTableReader::open(path)?;
        iters.push(SsTableIterator::new(reader)?);
    }

    let mut merge_iter = MergeIterator::new(iters)?;
    let mut outputs = Vec::new();

    let mut writer: Option<SsTableWriter> = None;
    let mut current_file_path = PathBuf::new();
    let mut current_file_size = 0;

    let mut last_user_key: Option<Vec<u8>> = None;
    let mut has_version_le_watermark = false;

    while let Some((key, value)) = merge_iter.next_entry()? {
        let is_new_key = match last_user_key {
            None => true,
            Some(ref last) => last != &key.user_key,
        };

        if is_new_key {
            last_user_key = Some(key.user_key.clone());
            has_version_le_watermark = false;
        }

        let keep = if key.ts >= watermark {
            if key.ts == watermark {
                has_version_le_watermark = true;
            }
            true
        } else if !has_version_le_watermark {
            has_version_le_watermark = true;
            !(value.is_none() && is_last_level)
        } else {
            false
        };

        if keep {
            if writer.is_none() {
                let id = next_file_id();
                current_file_path = output_dir.join(format!("{}.sst", id));
                writer = Some(SsTableWriter::new(&current_file_path)?);
                current_file_size = 0;
            }

            let w = writer.as_mut().unwrap();
            let key_len = key.user_key.len();
            let val_len = value.as_ref().map(|v| v.len()).unwrap_or(0);

            w.add(key, value.as_deref())?;
            current_file_size += 21 + key_len + val_len;

            if current_file_size >= max_file_size {
                let w = writer.take().unwrap();
                w.finish(fpr)?;
                outputs.push(current_file_path.clone());
            }
        }
    }

    if let Some(w) = writer.take() {
        w.finish(fpr)?;
        outputs.push(current_file_path);
    }

    Ok(outputs)
}
