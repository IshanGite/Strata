use crate::HlcTimestamp;
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct MemKey {
    pub user_key: Vec<u8>,
    pub ts: HlcTimestamp,
}

impl Ord for MemKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.user_key.cmp(&other.user_key) {
            std::cmp::Ordering::Equal => other.ts.cmp(&self.ts), // newest timestamp first
            ord => ord,
        }
    }
}

impl PartialOrd for MemKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct Node {
    key: MemKey,
    value: Option<Vec<u8>>,
    next: Vec<Option<usize>>,
}

pub struct SkipList {
    head: Vec<Option<usize>>,
    nodes: Vec<Node>,
    max_height: usize,
}

impl SkipList {
    pub fn new() -> Self {
        Self {
            head: vec![None; 32],
            nodes: Vec::new(),
            max_height: 1,
        }
    }

    fn random_height(&self) -> usize {
        let mut height = 1;
        while rand::random::<f32>() < 0.5 && height < 32 {
            height += 1;
        }
        height
    }

    pub fn insert(&mut self, key: MemKey, value: Option<Vec<u8>>) {
        let height = self.random_height();
        if height > self.max_height {
            self.max_height = height;
        }

        let mut update = vec![None; self.max_height];
        let mut curr_idx: Option<usize> = None;

        for level in (0..self.max_height).rev() {
            loop {
                let next_idx = match curr_idx {
                    None => self.head[level],
                    Some(idx) => self.nodes[idx].next[level],
                };

                match next_idx {
                    Some(n_idx) if self.nodes[n_idx].key < key => {
                        curr_idx = Some(n_idx);
                    }
                    _ => {
                        update[level] = curr_idx;
                        break;
                    }
                }
            }
        }

        let new_idx = self.nodes.len();
        self.nodes.push(Node {
            key,
            value,
            next: vec![None; height],
        });

        #[allow(clippy::needless_range_loop)]
        for level in 0..height {
            if let Some(pred_idx) = update[level] {
                self.nodes[new_idx].next[level] = self.nodes[pred_idx].next[level];
                self.nodes[pred_idx].next[level] = Some(new_idx);
            } else {
                self.nodes[new_idx].next[level] = self.head[level];
                self.head[level] = Some(new_idx);
            }
        }
    }

    pub fn find_greater_or_equal(&self, key: &MemKey) -> Option<usize> {
        let mut curr_idx: Option<usize> = None;
        for level in (0..self.max_height).rev() {
            loop {
                let next_idx = match curr_idx {
                    None => self.head[level],
                    Some(idx) => self.nodes[idx].next[level],
                };

                match next_idx {
                    Some(n_idx) if self.nodes[n_idx].key < *key => {
                        curr_idx = Some(n_idx);
                    }
                    _ => {
                        break;
                    }
                }
            }
        }

        match curr_idx {
            None => self.head[0],
            Some(idx) => self.nodes[idx].next[0],
        }
    }
}

impl Default for SkipList {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MemTable {
    inner: Arc<RwLock<SkipList>>,
    size: std::sync::atomic::AtomicUsize,
}

impl MemTable {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(SkipList::new())),
            size: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn put(&self, key: Vec<u8>, value: Vec<u8>, ts: HlcTimestamp) {
        let size_added = key.len() + value.len();
        let mem_key = MemKey { user_key: key, ts };
        self.inner.write().insert(mem_key, Some(value));
        self.size
            .fetch_add(size_added, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn delete(&self, key: Vec<u8>, ts: HlcTimestamp) {
        let size_added = key.len();
        let mem_key = MemKey { user_key: key, ts };
        self.inner.write().insert(mem_key, None);
        self.size
            .fetch_add(size_added, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn size(&self) -> usize {
        self.size.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn get(&self, key: &[u8], as_of: HlcTimestamp) -> Option<Option<Vec<u8>>> {
        let target = MemKey {
            user_key: key.to_vec(),
            ts: as_of,
        };
        let guard = self.inner.read();
        let idx = guard.find_greater_or_equal(&target)?;
        let node = &guard.nodes[idx];
        if node.key.user_key == key {
            Some(node.value.clone())
        } else {
            None
        }
    }

    pub fn scan(
        &self,
        start: &[u8],
        end: &[u8],
        as_of: HlcTimestamp,
    ) -> Vec<(Vec<u8>, Option<Vec<u8>>)> {
        let mut results = Vec::new();
        let guard = self.inner.read();
        let target = MemKey {
            user_key: start.to_vec(),
            ts: as_of,
        };

        let mut curr_idx = guard.find_greater_or_equal(&target);
        let mut last_seen_user_key: Option<Vec<u8>> = None;

        while let Some(idx) = curr_idx {
            let node = &guard.nodes[idx];
            let key = &node.key;

            if !end.is_empty() && key.user_key.as_slice() >= end {
                break;
            }

            if let Some(ref last) = last_seen_user_key {
                if last == &key.user_key {
                    curr_idx = node.next[0];
                    continue;
                }
            }

            if key.ts <= as_of {
                results.push((key.user_key.clone(), node.value.clone()));
                last_seen_user_key = Some(key.user_key.clone());
                curr_idx = node.next[0];
            } else {
                // Key is newer than as_of. Seek to (user_key, as_of)
                let seek_target = MemKey {
                    user_key: key.user_key.clone(),
                    ts: as_of,
                };
                curr_idx = guard.find_greater_or_equal(&seek_target);
            }
        }

        results
    }

    pub fn iter_all(&self) -> Vec<(MemKey, Option<Vec<u8>>)> {
        let guard = self.inner.read();
        let mut curr = guard.head[0];
        let mut res = Vec::new();
        while let Some(idx) = curr {
            let node = &guard.nodes[idx];
            res.push((node.key.clone(), node.value.clone()));
            curr = node.next[0];
        }
        res
    }

    pub fn clear(&self) {
        let mut guard = self.inner.write();
        guard.nodes.clear();
        guard.head = vec![None; 32];
        guard.max_height = 1;
        self.size.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Default for MemTable {
    fn default() -> Self {
        Self::new()
    }
}
