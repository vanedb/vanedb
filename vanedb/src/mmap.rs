use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::Path;

use memmap2::Mmap;

use crate::distance::{distance_fn, DistanceFn, DistanceMetric};
use crate::error::{Result, VaneError};
use crate::store::SearchResult;

const MAGIC: u32 = 0x564E4442; // "VNDB"
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 32;

fn metric_to_u32(m: DistanceMetric) -> u32 {
    match m {
        DistanceMetric::L2 => 0,
        DistanceMetric::Cosine => 1,
        DistanceMetric::Dot => 2,
    }
}

fn u32_to_metric(v: u32) -> Result<DistanceMetric> {
    match v {
        0 => Ok(DistanceMetric::L2),
        1 => Ok(DistanceMetric::Cosine),
        2 => Ok(DistanceMetric::Dot),
        _ => Err(VaneError::Io("invalid metric in file".to_string())),
    }
}

pub struct MmapVectorStoreBuilder {
    dim: usize,
    metric: DistanceMetric,
    ids: Vec<u64>,
    vectors: Vec<f32>,
    id_set: HashSet<u64>,
}

impl MmapVectorStoreBuilder {
    pub fn new(dim: usize, metric: DistanceMetric) -> Result<Self> {
        if dim == 0 {
            return Err(VaneError::EmptyVector);
        }
        Ok(Self {
            dim,
            metric,
            ids: Vec::new(),
            vectors: Vec::new(),
            id_set: HashSet::new(),
        })
    }

    pub fn add(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(VaneError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if self.id_set.contains(&id) {
            return Err(VaneError::DuplicateId { id });
        }
        self.ids.push(id);
        self.vectors.extend_from_slice(vector);
        self.id_set.insert(id);
        Ok(())
    }

    pub fn size(&self) -> usize {
        self.ids.len()
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        let mut f = fs::File::create(&tmp).map_err(|e| VaneError::Io(format!("create: {e}")))?;

        // Header
        f.write_all(&MAGIC.to_le_bytes())
            .map_err(|e| VaneError::Io(format!("write: {e}")))?;
        f.write_all(&VERSION.to_le_bytes())
            .map_err(|e| VaneError::Io(format!("write: {e}")))?;
        f.write_all(&(self.dim as u64).to_le_bytes())
            .map_err(|e| VaneError::Io(format!("write: {e}")))?;
        f.write_all(&(self.ids.len() as u64).to_le_bytes())
            .map_err(|e| VaneError::Io(format!("write: {e}")))?;
        f.write_all(&metric_to_u32(self.metric).to_le_bytes())
            .map_err(|e| VaneError::Io(format!("write: {e}")))?;
        f.write_all(&0u32.to_le_bytes())
            .map_err(|e| VaneError::Io(format!("write: {e}")))?; // reserved

        // IDs
        for &id in &self.ids {
            f.write_all(&id.to_le_bytes())
                .map_err(|e| VaneError::Io(format!("write: {e}")))?;
        }

        // Vectors
        for &v in &self.vectors {
            f.write_all(&v.to_le_bytes())
                .map_err(|e| VaneError::Io(format!("write: {e}")))?;
        }

        f.flush()
            .map_err(|e| VaneError::Io(format!("flush: {e}")))?;
        drop(f);

        fs::rename(&tmp, path).map_err(|e| VaneError::Io(format!("rename: {e}")))?;
        Ok(())
    }
}

pub struct MmapVectorStore {
    mmap: Mmap,
    dim: usize,
    num_vectors: usize,
    metric: DistanceMetric,
    dist_fn: DistanceFn,
    ids_offset: usize,
    vectors_offset: usize,
    id_map: HashMap<u64, usize>,
}

impl MmapVectorStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file =
            fs::File::open(path.as_ref()).map_err(|e| VaneError::Io(format!("open: {e}")))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| VaneError::Io(format!("mmap: {e}")))?;

        if mmap.len() < HEADER_SIZE {
            return Err(VaneError::Io("file too small".to_string()));
        }

        let magic = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(VaneError::Io("invalid magic".to_string()));
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(VaneError::Io(format!("unsupported version: {version}")));
        }

        let dim = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let num_vectors = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let metric_raw = u32::from_le_bytes(mmap[24..28].try_into().unwrap());
        let metric = u32_to_metric(metric_raw)?;

        if dim == 0 && num_vectors > 0 {
            return Err(VaneError::Io("zero dimension with vectors".to_string()));
        }

        let ids_size = num_vectors
            .checked_mul(8)
            .ok_or_else(|| VaneError::Io("size overflow".to_string()))?;
        let vecs_size = num_vectors
            .checked_mul(dim)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| VaneError::Io("size overflow".to_string()))?;
        let expected = HEADER_SIZE
            .checked_add(ids_size)
            .and_then(|n| n.checked_add(vecs_size))
            .ok_or_else(|| VaneError::Io("size overflow".to_string()))?;

        if mmap.len() < expected {
            return Err(VaneError::Io("file truncated".to_string()));
        }

        let ids_offset = HEADER_SIZE;
        let vectors_offset = HEADER_SIZE + ids_size;

        // Build ID → index map
        let mut id_map = HashMap::with_capacity(num_vectors);
        for i in 0..num_vectors {
            let off = ids_offset + i * 8;
            let id = u64::from_le_bytes(mmap[off..off + 8].try_into().unwrap());
            id_map.insert(id, i);
        }

        Ok(Self {
            mmap,
            dim,
            num_vectors,
            metric,
            dist_fn: distance_fn(metric),
            ids_offset,
            vectors_offset,
            id_map,
        })
    }

    pub fn size(&self) -> usize {
        self.num_vectors
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn contains(&self, id: u64) -> bool {
        self.id_map.contains_key(&id)
    }

    /// Get a vector by ID. Returns a slice into the memory-mapped file (zero-copy).
    pub fn get(&self, id: u64) -> Result<&[f32]> {
        let &idx = self.id_map.get(&id).ok_or(VaneError::NotFound { id })?;
        Ok(self.get_vec(idx))
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(VaneError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Err(VaneError::InvalidK);
        }

        let mut results: Vec<SearchResult> = (0..self.num_vectors)
            .map(|i| {
                let id = self.get_id(i);
                let vec = self.get_vec(i);
                SearchResult::new(id, (self.dist_fn)(query, vec))
            })
            .collect();

        results.sort();
        results.truncate(k);
        Ok(results)
    }

    fn get_id(&self, idx: usize) -> u64 {
        let off = self.ids_offset + idx * 8;
        u64::from_le_bytes(self.mmap[off..off + 8].try_into().unwrap())
    }

    /// Zero-copy vector access: reinterprets mmap'd bytes as f32 slice.
    fn get_vec(&self, idx: usize) -> &[f32] {
        let off = self.vectors_offset + idx * self.dim * 4;
        let bytes = &self.mmap[off..off + self.dim * 4];
        // SAFETY: f32 has alignment of 4, and mmap'd memory from the OS is page-aligned.
        // Data was written as native little-endian f32s.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.dim) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_add_and_size() {
        let mut b = MmapVectorStoreBuilder::new(3, DistanceMetric::L2).unwrap();
        b.add(1, &[1.0, 2.0, 3.0]).unwrap();
        b.add(2, &[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(b.size(), 2);
    }

    #[test]
    fn builder_rejects_wrong_dim() {
        let mut b = MmapVectorStoreBuilder::new(3, DistanceMetric::L2).unwrap();
        assert!(b.add(1, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn builder_rejects_duplicate() {
        let mut b = MmapVectorStoreBuilder::new(3, DistanceMetric::L2).unwrap();
        b.add(1, &[1.0, 2.0, 3.0]).unwrap();
        assert!(b.add(1, &[4.0, 5.0, 6.0]).is_err());
    }

    #[test]
    fn builder_rejects_zero_dim() {
        assert!(MmapVectorStoreBuilder::new(0, DistanceMetric::L2).is_err());
    }

    #[test]
    fn builder_save_creates_file() {
        let path = std::env::temp_dir().join("vanedb_test_mmap_builder.bin");
        let mut b = MmapVectorStoreBuilder::new(2, DistanceMetric::L2).unwrap();
        b.add(1, &[1.0, 2.0]).unwrap();
        b.save(&path).unwrap();
        assert!(path.exists());
        let meta = std::fs::metadata(&path).unwrap();
        // header(32) + 1 id(8) + 2 floats(8) = 48
        assert_eq!(meta.len(), 48);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn roundtrip_build_open_search() {
        let path = std::env::temp_dir().join("vanedb_test_mmap_roundtrip.bin");

        let mut b = MmapVectorStoreBuilder::new(3, DistanceMetric::L2).unwrap();
        b.add(10, &[0.0, 0.0, 0.0]).unwrap();
        b.add(20, &[1.0, 0.0, 0.0]).unwrap();
        b.add(30, &[10.0, 10.0, 10.0]).unwrap();
        b.save(&path).unwrap();

        let store = MmapVectorStore::open(&path).unwrap();
        assert_eq!(store.size(), 3);
        assert_eq!(store.dimension(), 3);
        assert!(store.contains(10));
        assert!(!store.contains(99));

        // Get (zero-copy)
        assert_eq!(store.get(10).unwrap(), &[0.0, 0.0, 0.0]);
        assert_eq!(store.get(20).unwrap(), &[1.0, 0.0, 0.0]);

        // Search
        let results = store.search(&[0.0, 0.1, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 10);
        assert_eq!(results[1].id, 20);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_bad_file() {
        let path = std::env::temp_dir().join("vanedb_test_mmap_bad.bin");
        std::fs::write(&path, b"garbage").unwrap();
        assert!(MmapVectorStore::open(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_truncated_file() {
        let path = std::env::temp_dir().join("vanedb_test_mmap_trunc.bin");
        let mut data = Vec::new();
        data.extend_from_slice(&MAGIC.to_le_bytes());
        data.extend_from_slice(&VERSION.to_le_bytes());
        data.extend_from_slice(&(3u64).to_le_bytes());
        data.extend_from_slice(&(1000u64).to_le_bytes());
        data.extend_from_slice(&(0u32).to_le_bytes());
        data.extend_from_slice(&(0u32).to_le_bytes());
        std::fs::write(&path, &data).unwrap();
        assert!(MmapVectorStore::open(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn search_wrong_dimension() {
        let path = std::env::temp_dir().join("vanedb_test_mmap_dim.bin");
        let mut b = MmapVectorStoreBuilder::new(3, DistanceMetric::L2).unwrap();
        b.add(1, &[1.0, 2.0, 3.0]).unwrap();
        b.save(&path).unwrap();

        let store = MmapVectorStore::open(&path).unwrap();
        assert!(store.search(&[1.0, 2.0], 1).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
