use crate::HlcTimestamp;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub struct Wal {
    file: File,
}

impl Wal {
    pub fn new(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        Ok(Self { file })
    }

    pub fn append(
        &mut self,
        is_delete: bool,
        key: &[u8],
        value: &[u8],
        ts: HlcTimestamp,
    ) -> io::Result<()> {
        let mut record = Vec::with_capacity(21 + key.len() + value.len() + 4);
        record.push(if is_delete { 1 } else { 0 });
        record.extend_from_slice(&ts.physical.to_le_bytes());
        record.extend_from_slice(&ts.logical.to_le_bytes());
        record.extend_from_slice(&(key.len() as u32).to_le_bytes());
        record.extend_from_slice(&(value.len() as u32).to_le_bytes());
        record.extend_from_slice(key);
        record.extend_from_slice(value);

        let crc = crc32fast::hash(&record);
        record.extend_from_slice(&crc.to_le_bytes());

        self.file.write_all(&record)?;
        Ok(())
    }

    pub fn replay<F>(&mut self, mut apply_fn: F) -> io::Result<()>
    where
        F: FnMut(bool, Vec<u8>, Vec<u8>, HlcTimestamp),
    {
        self.file.seek(SeekFrom::Start(0))?;
        let file_len = self.file.metadata()?.len();
        let mut current_offset = 0;

        loop {
            if current_offset >= file_len {
                break;
            }

            let mut header = [0u8; 21];
            let mut bytes_read = 0;
            let mut header_ok = true;
            while bytes_read < 21 {
                match self.file.read(&mut header[bytes_read..]) {
                    Ok(0) => {
                        header_ok = false;
                        break;
                    }
                    Ok(n) => bytes_read += n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }

            if !header_ok {
                self.file.set_len(current_offset)?;
                break;
            }

            let is_delete = header[0] == 1;
            let physical = u64::from_le_bytes(header[1..9].try_into().unwrap());
            let logical = u32::from_le_bytes(header[9..13].try_into().unwrap());
            let key_len = u32::from_le_bytes(header[13..17].try_into().unwrap()) as usize;
            let val_len = u32::from_le_bytes(header[17..21].try_into().unwrap()) as usize;

            let body_len = key_len + val_len + 4;
            let mut body = vec![0u8; body_len];
            let mut body_read = 0;
            let mut body_ok = true;
            while body_read < body_len {
                match self.file.read(&mut body[body_read..]) {
                    Ok(0) => {
                        body_ok = false;
                        break;
                    }
                    Ok(n) => body_read += n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }

            if !body_ok {
                self.file.set_len(current_offset)?;
                break;
            }

            let mut record = Vec::with_capacity(21 + key_len + val_len);
            record.extend_from_slice(&header);
            record.extend_from_slice(&body[..key_len + val_len]);

            let stored_crc = u32::from_le_bytes(body[key_len + val_len..].try_into().unwrap());
            let computed_crc = crc32fast::hash(&record);

            if stored_crc != computed_crc {
                self.file.set_len(current_offset)?;
                break;
            }

            let key = body[..key_len].to_vec();
            let value = body[key_len..key_len + val_len].to_vec();
            let ts = HlcTimestamp { physical, logical };
            apply_fn(is_delete, key, value, ts);

            current_offset += 21 + body_len as u64;
        }

        self.file.seek(SeekFrom::Start(current_offset))?;
        Ok(())
    }
}
