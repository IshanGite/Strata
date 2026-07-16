use crate::memtable::MemKey;
use crate::HlcTimestamp;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"STRATAS1";
const TARGET_BLOCK_SIZE: usize = 4096;

pub struct BloomFilter {
    pub bits: Vec<u8>,
    pub k: usize,
}

impl BloomFilter {
    pub fn new(keys: &[Vec<u8>], fpr: f32) -> Self {
        let n = keys.len();
        if n == 0 {
            return Self {
                bits: Vec::new(),
                k: 0,
            };
        }
        let ln2 = std::f32::consts::LN_2;
        let m = (-(n as f32) * fpr.ln() / (ln2 * ln2)).ceil() as usize;
        let num_bytes = m.div_ceil(8);
        let mut bits = vec![0u8; num_bytes];

        let k = ((m as f32 / n as f32) * ln2).round() as usize;
        let k = k.clamp(1, 30);

        for key in keys {
            for i in 0..k {
                let hash = Self::hash(key, i);
                let idx = (hash as usize) % (num_bytes * 8);
                bits[idx / 8] |= 1 << (idx % 8);
            }
        }

        Self { bits, k }
    }

    pub fn hash(key: &[u8], seed: usize) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(key);
        hasher.update(&seed.to_le_bytes());
        hasher.finalize()
    }

    pub fn contains(&self, key: &[u8]) -> bool {
        if self.bits.is_empty() {
            return false;
        }
        let num_bits = self.bits.len() * 8;
        for i in 0..self.k {
            let hash = Self::hash(key, i);
            let idx = (hash as usize) % num_bits;
            if (self.bits[idx / 8] & (1 << (idx % 8))) == 0 {
                return false;
            }
        }
        true
    }
}

pub struct IndexEntry {
    pub first_key: MemKey,
    pub offset: u64,
    pub size: u64,
}

pub struct SsTableWriter {
    file: File,
    offset: u64,
    index: Vec<IndexEntry>,
    current_block: Vec<u8>,
    block_first_key: Option<MemKey>,
    keys: Vec<Vec<u8>>,
    min_key: Option<MemKey>,
    max_key: Option<MemKey>,
}

impl SsTableWriter {
    pub fn new(path: &Path) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            file,
            offset: 0,
            index: Vec::new(),
            current_block: Vec::new(),
            block_first_key: None,
            keys: Vec::new(),
            min_key: None,
            max_key: None,
        })
    }

    pub fn add(&mut self, key: MemKey, value: Option<&[u8]>) -> io::Result<()> {
        if self.min_key.is_none() {
            self.min_key = Some(key.clone());
        }
        self.max_key = Some(key.clone());

        if !self.keys.contains(&key.user_key) {
            self.keys.push(key.user_key.clone());
        }

        if self.block_first_key.is_none() {
            self.block_first_key = Some(key.clone());
        }

        let is_delete = value.is_none();
        self.current_block.push(if is_delete { 1 } else { 0 });
        self.current_block
            .extend_from_slice(&key.ts.physical.to_le_bytes());
        self.current_block
            .extend_from_slice(&key.ts.logical.to_le_bytes());
        self.current_block
            .extend_from_slice(&(key.user_key.len() as u32).to_le_bytes());
        if let Some(val) = value {
            self.current_block
                .extend_from_slice(&(val.len() as u32).to_le_bytes());
            self.current_block.extend_from_slice(&key.user_key);
            self.current_block.extend_from_slice(val);
        } else {
            self.current_block.extend_from_slice(&0u32.to_le_bytes());
            self.current_block.extend_from_slice(&key.user_key);
        }

        if self.current_block.len() >= TARGET_BLOCK_SIZE {
            self.flush_block()?;
        }

        Ok(())
    }

    fn flush_block(&mut self) -> io::Result<()> {
        if self.current_block.is_empty() {
            return Ok(());
        }
        let size = self.current_block.len() as u64;
        self.file.write_all(&self.current_block)?;
        self.index.push(IndexEntry {
            first_key: self.block_first_key.take().unwrap(),
            offset: self.offset,
            size,
        });
        self.offset += size;
        self.current_block.clear();
        Ok(())
    }

    pub fn finish(mut self, fpr: f32) -> io::Result<()> {
        self.flush_block()?;

        // Write Sparse Index
        let index_offset = self.offset;
        let mut index_buf = Vec::new();
        index_buf.extend_from_slice(&(self.index.len() as u32).to_le_bytes());
        for entry in &self.index {
            index_buf.extend_from_slice(&entry.offset.to_le_bytes());
            index_buf.extend_from_slice(&entry.size.to_le_bytes());
            index_buf.extend_from_slice(&entry.first_key.ts.physical.to_le_bytes());
            index_buf.extend_from_slice(&entry.first_key.ts.logical.to_le_bytes());
            index_buf.extend_from_slice(&(entry.first_key.user_key.len() as u32).to_le_bytes());
            index_buf.extend_from_slice(&entry.first_key.user_key);
        }
        self.file.write_all(&index_buf)?;
        let index_size = index_buf.len() as u64;
        self.offset += index_size;

        // Build & Write Bloom Filter
        let bloom = BloomFilter::new(&self.keys, fpr);
        let bloom_offset = self.offset;
        self.file.write_all(&bloom.bits)?;
        let bloom_size = bloom.bits.len() as u64;
        self.offset += bloom_size;

        // Write Metadata Block
        let meta_offset = self.offset;
        let mut meta_buf = Vec::new();
        if let Some(ref min) = self.min_key {
            meta_buf.extend_from_slice(&(min.user_key.len() as u32).to_le_bytes());
            meta_buf.extend_from_slice(&min.user_key);
            meta_buf.extend_from_slice(&min.ts.physical.to_le_bytes());
            meta_buf.extend_from_slice(&min.ts.logical.to_le_bytes());
        } else {
            meta_buf.extend_from_slice(&0u32.to_le_bytes());
            meta_buf.extend_from_slice(&0u64.to_le_bytes());
            meta_buf.extend_from_slice(&0u32.to_le_bytes());
        }

        if let Some(ref max) = self.max_key {
            meta_buf.extend_from_slice(&(max.user_key.len() as u32).to_le_bytes());
            meta_buf.extend_from_slice(&max.user_key);
            meta_buf.extend_from_slice(&max.ts.physical.to_le_bytes());
            meta_buf.extend_from_slice(&max.ts.logical.to_le_bytes());
        } else {
            meta_buf.extend_from_slice(&0u32.to_le_bytes());
            meta_buf.extend_from_slice(&0u64.to_le_bytes());
            meta_buf.extend_from_slice(&0u32.to_le_bytes());
        }
        meta_buf.extend_from_slice(&(bloom.k as u32).to_le_bytes());
        self.file.write_all(&meta_buf)?;
        let meta_size = meta_buf.len() as u64;
        self.offset += meta_size;

        // Write Footer: 48 bytes
        let mut footer = Vec::with_capacity(48);
        footer.extend_from_slice(&index_offset.to_le_bytes());
        footer.extend_from_slice(&index_size.to_le_bytes());
        footer.extend_from_slice(&bloom_offset.to_le_bytes());
        footer.extend_from_slice(&bloom_size.to_le_bytes());
        footer.extend_from_slice(&meta_offset.to_le_bytes());
        footer.extend_from_slice(MAGIC);

        self.file.write_all(&footer)?;
        self.file.sync_all()?;
        Ok(())
    }
}

pub struct SsTableReader {
    file: File,
    pub index: Vec<IndexEntry>,
    pub bloom: BloomFilter,
    pub min_key: MemKey,
    pub max_key: MemKey,
}

impl SsTableReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len < 48 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "File too small for footer",
            ));
        }

        // Read footer
        let mut footer = [0u8; 48];
        file.seek(SeekFrom::Start(file_len - 48))?;
        file.read_exact(&mut footer)?;

        if &footer[40..48] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid magic number",
            ));
        }

        let index_offset = u64::from_le_bytes(footer[0..8].try_into().unwrap());
        let _index_size = u64::from_le_bytes(footer[8..16].try_into().unwrap());
        let bloom_offset = u64::from_le_bytes(footer[16..24].try_into().unwrap());
        let bloom_size = u64::from_le_bytes(footer[24..32].try_into().unwrap());
        let meta_offset = u64::from_le_bytes(footer[32..40].try_into().unwrap());

        // Read Metadata
        file.seek(SeekFrom::Start(meta_offset))?;
        let mut min_key_len_bytes = [0u8; 4];
        file.read_exact(&mut min_key_len_bytes)?;
        let min_key_len = u32::from_le_bytes(min_key_len_bytes) as usize;
        let mut min_user_key = vec![0u8; min_key_len];
        file.read_exact(&mut min_user_key)?;
        let mut min_ts_phys = [0u8; 8];
        file.read_exact(&mut min_ts_phys)?;
        let mut min_ts_log = [0u8; 4];
        file.read_exact(&mut min_ts_log)?;
        let min_key = MemKey {
            user_key: min_user_key,
            ts: HlcTimestamp {
                physical: u64::from_le_bytes(min_ts_phys),
                logical: u32::from_le_bytes(min_ts_log),
            },
        };

        let mut max_key_len_bytes = [0u8; 4];
        file.read_exact(&mut max_key_len_bytes)?;
        let max_key_len = u32::from_le_bytes(max_key_len_bytes) as usize;
        let mut max_user_key = vec![0u8; max_key_len];
        file.read_exact(&mut max_user_key)?;
        let mut max_ts_phys = [0u8; 8];
        file.read_exact(&mut max_ts_phys)?;
        let mut max_ts_log = [0u8; 4];
        file.read_exact(&mut max_ts_log)?;
        let max_key = MemKey {
            user_key: max_user_key,
            ts: HlcTimestamp {
                physical: u64::from_le_bytes(max_ts_phys),
                logical: u32::from_le_bytes(max_ts_log),
            },
        };

        let mut bloom_k_bytes = [0u8; 4];
        file.read_exact(&mut bloom_k_bytes)?;
        let bloom_k = u32::from_le_bytes(bloom_k_bytes) as usize;

        // Read Bloom Filter
        let mut bloom_bits = vec![0u8; bloom_size as usize];
        file.seek(SeekFrom::Start(bloom_offset))?;
        file.read_exact(&mut bloom_bits)?;
        let bloom = BloomFilter {
            bits: bloom_bits,
            k: bloom_k,
        };

        // Read Sparse Index
        file.seek(SeekFrom::Start(index_offset))?;
        let mut num_entries_bytes = [0u8; 4];
        file.read_exact(&mut num_entries_bytes)?;
        let num_entries = u32::from_le_bytes(num_entries_bytes) as usize;
        let mut index = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            let mut off_bytes = [0u8; 8];
            file.read_exact(&mut off_bytes)?;
            let offset = u64::from_le_bytes(off_bytes);
            let mut sz_bytes = [0u8; 8];
            file.read_exact(&mut sz_bytes)?;
            let size = u64::from_le_bytes(sz_bytes);

            let mut phys_bytes = [0u8; 8];
            file.read_exact(&mut phys_bytes)?;
            let mut log_bytes = [0u8; 4];
            file.read_exact(&mut log_bytes)?;
            let mut key_len_bytes = [0u8; 4];
            file.read_exact(&mut key_len_bytes)?;
            let key_len = u32::from_le_bytes(key_len_bytes) as usize;
            let mut user_key = vec![0u8; key_len];
            file.read_exact(&mut user_key)?;

            index.push(IndexEntry {
                first_key: MemKey {
                    user_key,
                    ts: HlcTimestamp {
                        physical: u64::from_le_bytes(phys_bytes),
                        logical: u32::from_le_bytes(log_bytes),
                    },
                },
                offset,
                size,
            });
        }

        Ok(Self {
            file,
            index,
            bloom,
            min_key,
            max_key,
        })
    }

    pub fn get_from_block(
        &mut self,
        block_idx: usize,
        key: &[u8],
        as_of: HlcTimestamp,
    ) -> io::Result<Option<Option<Vec<u8>>>> {
        let entry = &self.index[block_idx];
        let mut block_data = vec![0u8; entry.size as usize];
        self.file.seek(SeekFrom::Start(entry.offset))?;
        self.file.read_exact(&mut block_data)?;

        let mut offset = 0;
        let mut best_val: Option<Option<Vec<u8>>> = None;

        while offset < block_data.len() {
            let is_delete = block_data[offset] == 1;
            let physical =
                u64::from_le_bytes(block_data[offset + 1..offset + 9].try_into().unwrap());
            let logical =
                u32::from_le_bytes(block_data[offset + 9..offset + 13].try_into().unwrap());
            let key_len =
                u32::from_le_bytes(block_data[offset + 13..offset + 17].try_into().unwrap())
                    as usize;
            let val_len =
                u32::from_le_bytes(block_data[offset + 17..offset + 21].try_into().unwrap())
                    as usize;

            let cur_key = &block_data[offset + 21..offset + 21 + key_len];

            if cur_key == key {
                let ts = HlcTimestamp { physical, logical };
                if ts <= as_of {
                    let value = if is_delete {
                        None
                    } else {
                        Some(
                            block_data[offset + 21 + key_len..offset + 21 + key_len + val_len]
                                .to_vec(),
                        )
                    };
                    // Since entries inside a block are sorted newest first, the first version we match
                    // that is <= as_of is guaranteed to be the newest valid version <= as_of.
                    if best_val.is_none() {
                        best_val = Some(value);
                        break;
                    }
                }
            }

            offset += 21 + key_len + val_len;
        }

        Ok(best_val)
    }

    pub fn get(&mut self, key: &[u8], as_of: HlcTimestamp) -> io::Result<Option<Option<Vec<u8>>>> {
        if !self.bloom.contains(key) {
            return Ok(None);
        }

        // Fast range check using min/max keys
        if key < self.min_key.user_key.as_slice() || key > self.max_key.user_key.as_slice() {
            return Ok(None);
        }

        let mut candidate_idx = None;
        for (idx, entry) in self.index.iter().enumerate() {
            let u_cmp = entry.first_key.user_key.as_slice().cmp(key);
            if u_cmp == std::cmp::Ordering::Less {
                candidate_idx = Some(idx);
            } else if u_cmp == std::cmp::Ordering::Equal {
                if entry.first_key.ts >= as_of {
                    candidate_idx = Some(idx);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        let start_idx = candidate_idx.unwrap_or(0);

        for idx in start_idx..std::cmp::min(start_idx + 2, self.index.len()) {
            let entry = &self.index[idx];
            let u_cmp = entry.first_key.user_key.as_slice().cmp(key);
            if u_cmp == std::cmp::Ordering::Less || u_cmp == std::cmp::Ordering::Equal {
                let res = self.get_from_block(idx, key, as_of)?;
                if res.is_some() {
                    return Ok(res);
                }
            }
        }

        Ok(None)
    }

    pub fn read_block_entries(
        &mut self,
        offset: u64,
        size: u64,
    ) -> io::Result<Vec<(MemKey, Option<Vec<u8>>)>> {
        let mut block_data = vec![0u8; size as usize];
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut block_data)?;

        let mut offset = 0;
        let mut entries = Vec::new();

        while offset < block_data.len() {
            let is_delete = block_data[offset] == 1;
            let physical =
                u64::from_le_bytes(block_data[offset + 1..offset + 9].try_into().unwrap());
            let logical =
                u32::from_le_bytes(block_data[offset + 9..offset + 13].try_into().unwrap());
            let key_len =
                u32::from_le_bytes(block_data[offset + 13..offset + 17].try_into().unwrap())
                    as usize;
            let val_len =
                u32::from_le_bytes(block_data[offset + 17..offset + 21].try_into().unwrap())
                    as usize;

            let user_key = block_data[offset + 21..offset + 21 + key_len].to_vec();
            let value = if is_delete {
                None
            } else {
                Some(block_data[offset + 21 + key_len..offset + 21 + key_len + val_len].to_vec())
            };

            entries.push((
                MemKey {
                    user_key,
                    ts: HlcTimestamp { physical, logical },
                },
                value,
            ));

            offset += 21 + key_len + val_len;
        }

        Ok(entries)
    }
}
