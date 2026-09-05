//! Exact search over a memory-mapped file.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::distance::{self as d, Metric};
use crate::error::{Result, VaneError};
use crate::store::SearchResult;
use crate::validation::validate_finite;

const MAGIC: u32 = 0x564E4442; // "VNDB"
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 32;

/// Write buffer for [`DiskStoreBuilder::save`]. Ids and vectors are
/// encoded element-wise to keep the on-disk layout explicitly little-endian;
/// unbuffered that cost one `write` syscall per element, so a 10k x 128 store
/// issued 1.29M of them.
const WRITE_BUFFER_BYTES: usize = 64 * 1024;

fn metric_to_u32(m: Metric) -> u32 {
    match m {
        Metric::L2 => 0,
        Metric::Cosine => 1,
        Metric::Dot => 2,
    }
}

fn u32_to_metric(v: u32) -> Result<Metric> {
    match v {
        0 => Ok(Metric::L2),
        1 => Ok(Metric::Cosine),
        2 => Ok(Metric::Dot),
        _ => Err(VaneError::Io("invalid metric in file".to_string())),
    }
}

/// Collects vectors and writes them to a file [`DiskStore`] can open.
///
/// Vectors are held in memory until [`save`](Self::save); the memory saving
/// is on the reading side.
pub struct DiskStoreBuilder {
    dim: usize,
    metric: Metric,
    ids: Vec<u64>,
    vectors: Vec<f32>,
    id_set: HashSet<u64>,
}

impl DiskStoreBuilder {
    /// Starts a store for vectors of `dim` components.
    pub fn new(dim: usize, metric: Metric) -> Result<Self> {
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

    /// Adds `vector` under `id`.
    ///
    /// Fails if `id` is taken, if the length differs from `dim`, or if any
    /// component is not finite.
    pub fn add(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(VaneError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        validate_finite(vector, "vector")?;
        if self.id_set.contains(&id) {
            return Err(VaneError::DuplicateId { id });
        }
        self.ids.push(id);
        self.vectors.extend_from_slice(vector);
        self.id_set.insert(id);
        Ok(())
    }

    /// Component count of every vector this builder accepts.
    ///
    /// The C++ `DiskStoreBuilder` exposes the same accessor.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Number of vectors collected so far.
    /// Number of vectors in the mapped file.
    pub fn size(&self) -> usize {
        self.ids.len()
    }

    /// Writes the store to `path`.
    ///
    /// The file is built beside the destination and renamed into place after
    /// an fsync, so an interrupted write cannot leave a half-written store
    /// where a reader would find it. The layout is little-endian and shared
    /// with the C++ implementation.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let temp = crate::atomic_write::AtomicFile::new(path);
        let file =
            fs::File::create(temp.path()).map_err(|e| VaneError::Io(format!("create: {e}")))?;
        let mut f = BufWriter::with_capacity(WRITE_BUFFER_BYTES, file);

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

        // into_inner flushes the buffer; the fsync below must see every byte.
        let f = f
            .into_inner()
            .map_err(|e| VaneError::Io(format!("flush: {e}")))?;
        // Durability: fsync data + metadata before rename so a crash mid-write
        // can't leave a half-written file in place. Mirrors fsync_file in
        // vanedb-cpp src/core/detail/file_utils.h.
        f.sync_all()
            .map_err(|e| VaneError::Io(format!("sync: {e}")))?;
        drop(f);

        temp.commit(path)
    }
}

/// Exact k-nearest-neighbour search over a memory-mapped file.
///
/// Vectors stay on disk and are paged in by the kernel as the scan touches
/// them, so a corpus larger than RAM remains searchable. Read-only; build
/// one with [`DiskStoreBuilder`].
pub struct DiskStore {
    mmap: Mmap,
    dim: usize,
    num_vectors: usize,
    metric: Metric,
    ids_offset: usize,
    vectors_offset: usize,
    id_map: HashMap<u64, usize>,
}

impl DiskStore {
    /// Maps the store at `path`.
    ///
    /// Validates the header, checks every stored component is finite, and
    /// builds the id index, so this is linear in the corpus rather than a
    /// constant-cost mapping. A corrupt or truncated file is rejected here
    /// rather than surfacing as a wrong answer later.
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

        for offset in (vectors_offset..expected).step_by(4) {
            if !f32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap()).is_finite() {
                return Err(VaneError::Io(
                    "corrupted file: vector values must be finite".to_string(),
                ));
            }
        }

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
            ids_offset,
            vectors_offset,
            id_map,
        })
    }

    /// Number of vectors in the mapped file.
    pub fn size(&self) -> usize {
        self.num_vectors
    }

    /// Component count of every vector in this store.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// The metric recorded in the file.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Whether a vector is stored under `id`.
    pub fn contains(&self, id: u64) -> bool {
        self.id_map.contains_key(&id)
    }

    /// Get a vector by ID. Returns a slice into the memory-mapped file (zero-copy).
    pub fn get(&self, id: u64) -> Result<&[f32]> {
        let &idx = self.id_map.get(&id).ok_or(VaneError::NotFound { id })?;
        Ok(self.get_vec(idx))
    }

    /// The `k` nearest vectors to `query`, nearest first.
    ///
    /// Returns fewer than `k` results when the file holds fewer vectors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(VaneError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        validate_finite(query, "query")?;
        if k == 0 {
            return Err(VaneError::InvalidK);
        }

        // Monomorphized per-metric scan + top-k selection instead of a full
        // sort through the dist_fn pointer — same treatment as
        // Store::search (O(n log n) -> O(n + k log k)).
        macro_rules! scan {
            ($dist:path) => {
                (0..self.num_vectors)
                    .map(|i| SearchResult::new(self.get_id(i), $dist(query, self.get_vec(i))))
                    .collect()
            };
        }
        let mut results: Vec<SearchResult> = match self.metric {
            Metric::L2 => scan!(d::l2_squared),
            Metric::Cosine => scan!(d::cosine_distance),
            Metric::Dot => scan!(d::dot_distance),
        };

        if k < results.len() {
            results.select_nth_unstable(k - 1);
            results.truncate(k);
        }
        results.sort_unstable();
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
        let mut b = DiskStoreBuilder::new(3, Metric::L2).unwrap();
        b.add(1, &[1.0, 2.0, 3.0]).unwrap();
        b.add(2, &[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(b.size(), 2);
    }

    #[test]
    fn builder_rejects_wrong_dim() {
        let mut b = DiskStoreBuilder::new(3, Metric::L2).unwrap();
        assert!(b.add(1, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn builder_rejects_duplicate() {
        let mut b = DiskStoreBuilder::new(3, Metric::L2).unwrap();
        b.add(1, &[1.0, 2.0, 3.0]).unwrap();
        assert!(b.add(1, &[4.0, 5.0, 6.0]).is_err());
    }

    #[test]
    fn builder_rejects_zero_dim() {
        assert!(DiskStoreBuilder::new(0, Metric::L2).is_err());
    }

    #[test]
    fn builder_save_creates_file() {
        let path = std::env::temp_dir().join("vanedb_test_mmap_builder.bin");
        let mut b = DiskStoreBuilder::new(2, Metric::L2).unwrap();
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

        let mut b = DiskStoreBuilder::new(3, Metric::L2).unwrap();
        b.add(10, &[0.0, 0.0, 0.0]).unwrap();
        b.add(20, &[1.0, 0.0, 0.0]).unwrap();
        b.add(30, &[10.0, 10.0, 10.0]).unwrap();
        b.save(&path).unwrap();

        let store = DiskStore::open(&path).unwrap();
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
        assert!(DiskStore::open(&path).is_err());
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
        assert!(DiskStore::open(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn search_wrong_dimension() {
        let path = std::env::temp_dir().join("vanedb_test_mmap_dim.bin");
        let mut b = DiskStoreBuilder::new(3, Metric::L2).unwrap();
        b.add(1, &[1.0, 2.0, 3.0]).unwrap();
        b.save(&path).unwrap();

        let store = DiskStore::open(&path).unwrap();
        assert!(store.search(&[1.0, 2.0], 1).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
