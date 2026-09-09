//! Exact search over a memory-mapped file.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::distance::{self as d, Metric};
use crate::error::{Result, VaneError};
use crate::flat::SearchResult;
use crate::validation::{validate_query, validate_vector};

/// Literal `VNDB` as the first four bytes on disk, matching the contract in
/// `conformance/README.md` and the C++ engine's `DiskIndex::MAGIC`. Built from
/// the bytes rather than hand-written hex: the previous constant claimed
/// "VNDB" in a comment and actually wrote `BDNV`.
const MAGIC: u32 = u32::from_le_bytes(*b"VNDB");
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 32;

/// Write buffer for [`DiskIndexBuilder::save`]. Ids and vectors are
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
        _ => Err(VaneError::corrupt("invalid metric in file")),
    }
}

/// Collects vectors and writes them to a file [`DiskIndex`] can open.
///
/// Vectors are held in memory until [`save`](Self::save); the memory saving
/// is on the reading side. This is the only way to produce a file
/// [`DiskIndex::open`] accepts.
///
/// # Examples
///
/// Build a file, then map it for search. The two halves are usually separate
/// programs — build once, then open the file read-only from as many processes
/// as you like.
///
/// ```
/// use vanedb::{DiskIndex, DiskIndexBuilder, Metric};
///
/// # fn main() -> vanedb::Result<()> {
/// let path = std::env::temp_dir().join(format!("vanedb-doc-example-{}.vndb", std::process::id()));
///
/// let mut builder = DiskIndexBuilder::new(3, Metric::L2)?;
/// builder.add(1, &[1.0, 0.0, 0.0])?;
/// builder.add(2, &[0.0, 1.0, 0.0])?;
/// builder.save(&path)?;
///
/// // SAFETY: nothing else writes or truncates this file while it is mapped.
/// // `save` renames a temporary into place, so rebuilding the path while a
/// // reader holds the old inode open is safe; editing in place is not.
/// let index = unsafe { DiskIndex::open(&path)? };
/// assert_eq!(index.len(), 2);
///
/// let hits = index.search(&[0.9, 0.1, 0.0], 1)?;
/// assert_eq!(hits[0].id, 1);
///
/// # std::fs::remove_file(&path).ok();
/// # Ok(())
/// # }
/// ```
pub struct DiskIndexBuilder {
    dim: usize,
    metric: Metric,
    ids: Vec<u64>,
    vectors: Vec<f32>,
    id_set: HashSet<u64>,
}

impl std::fmt::Debug for DiskIndexBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskIndexBuilder")
            .field("dim", &self.dim)
            .field("metric", &self.metric)
            .field("len", &self.ids.len())
            .finish()
    }
}

impl DiskIndexBuilder {
    /// Starts a store for vectors of `dim` components.
    ///
    /// # Errors
    ///
    /// Returns [`VaneError::ZeroDimension`] if `dim` is zero.
    pub fn new(dim: usize, metric: Metric) -> Result<Self> {
        if dim == 0 {
            return Err(VaneError::ZeroDimension);
        }
        // Matches FlatIndex::new and ApproxIndexBuilder::build: a dimension
        // that cannot be sized in bytes is rejected here rather than
        // overflowing later, where it surfaces as a panic.
        if dim.checked_mul(std::mem::size_of::<f32>()).is_none() {
            return Err(VaneError::InvalidParameter(
                "dim * size_of::<f32>() overflows usize",
            ));
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
        validate_vector(vector, self.dim)?;
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
    /// The C++ `DiskIndexBuilder` exposes the same accessor.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Number of vectors collected so far.
    pub fn size(&self) -> usize {
        self.ids.len()
    }

    /// Number of vectors collected so far. Same count as [`size`](Self::size);
    /// both spellings exist so a program is not tied to one engine (#85).
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether nothing has been added yet.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Writes the store to `path`.
    ///
    /// The file is built beside the destination and renamed into place after
    /// an fsync, so an interrupted write cannot leave a half-written store
    /// where a reader would find it. The layout is little-endian and shared
    /// with the C++ implementation, which reads and writes the same bytes
    /// (bench/tests/cross_engine_format.rs).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let temp = crate::atomic_write::AtomicFile::new(path);
        let file = fs::File::create(temp.path()).map_err(|e| VaneError::from_io("create", e))?;
        let mut f = BufWriter::with_capacity(WRITE_BUFFER_BYTES, file);

        // Header
        f.write_all(&MAGIC.to_le_bytes())
            .map_err(|e| VaneError::from_io("write", e))?;
        f.write_all(&VERSION.to_le_bytes())
            .map_err(|e| VaneError::from_io("write", e))?;
        f.write_all(&(self.dim as u64).to_le_bytes())
            .map_err(|e| VaneError::from_io("write", e))?;
        f.write_all(&(self.ids.len() as u64).to_le_bytes())
            .map_err(|e| VaneError::from_io("write", e))?;
        f.write_all(&metric_to_u32(self.metric).to_le_bytes())
            .map_err(|e| VaneError::from_io("write", e))?;
        f.write_all(&0u32.to_le_bytes())
            .map_err(|e| VaneError::from_io("write", e))?; // reserved

        // IDs
        for &id in &self.ids {
            f.write_all(&id.to_le_bytes())
                .map_err(|e| VaneError::from_io("write", e))?;
        }

        // Vectors
        for &v in &self.vectors {
            f.write_all(&v.to_le_bytes())
                .map_err(|e| VaneError::from_io("write", e))?;
        }

        // into_inner flushes the buffer; the fsync below must see every byte.
        // into_inner yields an IntoInnerError wrapping the writer; take the
        // io::Error out of it so the classification and source chain work.
        let f = f
            .into_inner()
            .map_err(|e| VaneError::from_io("flush", e.into_error()))?;
        // Durability: fsync data + metadata before rename so a crash mid-write
        // can't leave a half-written file in place. Mirrors fsync_file in
        // vanedb-cpp src/core/detail/file_utils.h.
        f.sync_all().map_err(|e| VaneError::from_io("sync", e))?;
        drop(f);

        temp.commit(path)
    }
}

/// Exact k-nearest-neighbour search over a memory-mapped file.
///
/// Vectors stay on disk and are paged in by the kernel as the scan touches
/// them, so a corpus larger than RAM remains searchable. Read-only; build
/// one with [`DiskIndexBuilder`].
pub struct DiskIndex {
    mmap: Mmap,
    dim: usize,
    num_vectors: usize,
    metric: Metric,
    ids_offset: usize,
    vectors_offset: usize,
    id_map: HashMap<u64, usize>,
}

impl std::fmt::Debug for DiskIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskIndex")
            .field("dim", &self.dim)
            .field("metric", &self.metric)
            .field("len", &self.num_vectors)
            .finish()
    }
}

impl DiskIndex {
    /// Maps the store at `path`.
    ///
    /// Validates the header, checks every stored component is finite, and
    /// builds the id index, so this is linear in the corpus rather than a
    /// constant-cost mapping.
    ///
    /// # Integrity
    ///
    /// What this rejects: a bad magic or version, a header whose declared
    /// geometry disagrees with the file length in either direction, a
    /// dimension too large to address, a non-finite stored component, a
    /// duplicate id, and a nonzero reserved word.
    ///
    /// What it cannot reject — the format carries no checksum, so all of the
    /// following are structurally valid files:
    ///
    /// - A flipped bit inside a stored id, returned as data. A flip inside a
    ///   vector component is usually returned too, though one that produces an
    ///   infinity or a NaN is caught by the finiteness check.
    /// - A flipped `metric`, when the new value is another defined metric.
    ///   Nothing records which was intended, so every distance changes and the
    ///   file still loads.
    /// - A *coordinated* rewrite of `dim` and `num_vectors` that preserves the
    ///   total length. `expected` is `32 + n * (8 + 4 * dim)`, so any pair on
    ///   that curve passes: a 2592-byte file written as `dim = 8, n = 64` also
    ///   reads as `dim = 6, n = 80`, at the wrong stride. The length check
    ///   makes a *single-bit* flip in either field almost always detectable —
    ///   it moves the length — but it cannot pin the geometry.
    /// - Any `dim` at all when `num_vectors` is 0, since the file is then
    ///   header-only whatever the dimension. No vector can be wrong, but
    ///   `dimension` will report the corrupted value.
    ///
    /// A caller who needs integrity against bitrot must supply it — verify a
    /// digest of the file before opening it, or store on a filesystem that
    /// checksums blocks.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the underlying file's contents and length
    /// cannot change, in this process or another, from before this call until
    /// the returned index is dropped. Validation does not establish this
    /// guarantee: the index borrows directly from mapped bytes, and modifying
    /// them can cause undefined behavior. Truncation can also cause a process
    /// fault such as `SIGBUS`, rather than a recoverable error.
    ///
    /// Writing with [`DiskIndexBuilder::save`] is safe against this: it builds
    /// a temporary sibling and renames it into place, which replaces the
    /// directory entry and leaves this mapping pointing at the old, intact
    /// inode. Rebuilding an index while readers hold it open is therefore
    /// fine. Editing a mapped file in place is not.
    pub unsafe fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = fs::File::open(path.as_ref()).map_err(|e| VaneError::from_io("open", e))?;
        // SAFETY: the caller guarantees immutability for this mapping's lifetime.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| VaneError::from_io("mmap", e))?;

        if mmap.len() < HEADER_SIZE {
            return Err(VaneError::corrupt("file too small"));
        }

        let magic = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(VaneError::corrupt("invalid magic"));
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(VaneError::corrupt(format!(
                "unsupported version: {version}"
            )));
        }

        // `try_from`, not `as`. `as` truncates on a 32-bit target, so a header
        // declaring `dim = 0x1_0000_0001` would read as 1 there and be
        // rejected on a 64-bit host — the two would disagree about the same
        // file, which is exactly what `conformance/README.md` forbids and what
        // `both_engines_accept_and_reject_exactly_the_same_files` asserts. The
        // v2 reader already does this (`graph_format.rs`, "graph size exceeds
        // this platform"). Not reachable on a target built today — wasm32 does
        // not enable `disk` — which is why it is a guard rather than a fix.
        let header_size = |bytes: [u8; 8]| {
            usize::try_from(u64::from_le_bytes(bytes))
                .map_err(|_| VaneError::corrupt("header size exceeds this platform"))
        };
        let dim = header_size(mmap[8..16].try_into().unwrap())?;
        let num_vectors = header_size(mmap[16..24].try_into().unwrap())?;
        let metric_raw = u32::from_le_bytes(mmap[24..28].try_into().unwrap());
        let metric = u32_to_metric(metric_raw)?;
        // Offsets 28..32 are reserved and specified as zero. Rejecting a
        // nonzero value is what keeps them claimable: a reader that ignores
        // them can never be given a meaning later, because every binary
        // already in the field would silently misread a file that used one.
        if u32::from_le_bytes(mmap[28..32].try_into().unwrap()) != 0 {
            return Err(VaneError::corrupt("reserved header bytes are not zero"));
        }

        if dim == 0 && num_vectors > 0 {
            return Err(VaneError::corrupt("zero dimension with vectors"));
        }

        // Bound `dim` on its own, not only through the product below. When
        // `num_vectors` is 0 the product is 0 for any `dim`, so the length
        // check cannot see the field at all and an empty store would accept a
        // `dim` of 2^62 — which the C++ reader rejects here, making one file
        // the two engines disagree about. A dimension whose vector cannot be
        // addressed is unreadable whether or not any vector is stored.
        if dim > usize::MAX / 4 {
            return Err(VaneError::corrupt("dimension overflows a byte count"));
        }

        let ids_size = num_vectors
            .checked_mul(8)
            .ok_or_else(|| VaneError::corrupt("size overflow"))?;
        let vecs_size = num_vectors
            .checked_mul(dim)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| VaneError::corrupt("size overflow"))?;
        let expected = HEADER_SIZE
            .checked_add(ids_size)
            .and_then(|n| n.checked_add(vecs_size))
            .ok_or_else(|| VaneError::corrupt("size overflow"))?;

        // Equality, not `>=`. `expected` is derived from the header, so a
        // one-sided check lets a header that understates the geometry move the
        // goalpost instead of tripping the guard: flipping one bit of `dim`
        // from 3 to 2 shrinks `expected` below the real length, and the payload
        // is then read at the wrong stride, with `get` returning a vector that
        // straddles two stored records. `save` writes header + ids + vectors
        // and nothing else, so any file this crate wrote matches exactly. The
        // graph reader already holds this line (`graph_format.rs`, "trailing
        // bytes in VNDB graph").
        if mmap.len() != expected {
            return Err(VaneError::corrupt(if mmap.len() < expected {
                "file truncated"
            } else {
                "file longer than its header declares"
            }));
        }

        let ids_offset = HEADER_SIZE;
        let vectors_offset = HEADER_SIZE + ids_size;

        for offset in (vectors_offset..expected).step_by(4) {
            if !f32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap()).is_finite() {
                return Err(VaneError::corrupt(
                    "corrupted file: vector values must be finite",
                ));
            }
        }

        // Build ID → index map. Duplicates would silently overwrite, leaving
        // `size()` disagreeing with the map, `get` returning a different row
        // than the id names, and `search` emitting one id twice. The HNSW
        // loader enforces the same bijection (approx/persistence.rs).
        let mut id_map = HashMap::with_capacity(num_vectors);
        for i in 0..num_vectors {
            let off = ids_offset + i * 8;
            let id = u64::from_le_bytes(mmap[off..off + 8].try_into().unwrap());
            if id_map.insert(id, i).is_some() {
                return Err(VaneError::corrupt("duplicate vector id"));
            }
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

    /// Number of vectors in the mapped file. Same count as
    /// [`size`](Self::size); both spellings exist so a program is not tied
    /// to one engine (#85).
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Whether the mapped file holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
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

    /// The vector stored under `id`.
    ///
    /// Borrows directly from the mapping today, so this copies nothing. The
    /// return type is [`Cow`] rather than `&[f32]` because a slice would make
    /// the on-disk layout part of the signature: a store that encodes vectors
    /// in any form other than native `f32` has nothing to lend and must decode
    /// into a buffer. Callers that want an owned vector can use
    /// [`Cow::into_owned`].
    pub fn get(&self, id: u64) -> Result<Cow<'_, [f32]>> {
        let &idx = self.id_map.get(&id).ok_or(VaneError::NotFound { id })?;
        Ok(Cow::Borrowed(self.get_vec(idx)))
    }

    /// The `k` nearest vectors to `query`, nearest first.
    ///
    /// Returns fewer than `k` results when the file holds fewer vectors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        validate_query(query, self.dim, k)?;

        // Monomorphized per-metric scan + top-k selection instead of a full
        // sort through the dist_fn pointer — same treatment as
        // FlatIndex::search (O(n log n) -> O(n + k log k)).
        macro_rules! scan {
            ($dist:path) => {
                // Bounded top-k over the stream; see flat/topk.rs.
                crate::flat::topk::select(
                    (0..self.num_vectors)
                        .map(|i| SearchResult::new(self.get_id(i), $dist(query, self.get_vec(i)))),
                    k,
                )
            };
        }
        let results = match self.metric {
            Metric::L2 => scan!(d::l2_squared),
            Metric::Cosine => scan!(d::cosine_distance),
            Metric::Dot => scan!(d::dot_distance),
        };
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
        let mut b = DiskIndexBuilder::new(3, Metric::L2).unwrap();
        b.add(1, &[1.0, 2.0, 3.0]).unwrap();
        b.add(2, &[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(b.size(), 2);
    }

    #[test]
    fn builder_rejects_wrong_dim() {
        let mut b = DiskIndexBuilder::new(3, Metric::L2).unwrap();
        assert!(b.add(1, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn builder_rejects_duplicate() {
        let mut b = DiskIndexBuilder::new(3, Metric::L2).unwrap();
        b.add(1, &[1.0, 2.0, 3.0]).unwrap();
        assert!(b.add(1, &[4.0, 5.0, 6.0]).is_err());
    }

    #[test]
    fn builder_rejects_zero_dim() {
        assert!(DiskIndexBuilder::new(0, Metric::L2).is_err());
    }

    #[test]
    fn builder_save_creates_file() {
        let path = std::env::temp_dir().join(format!(
            "vanedb_test_mmap_builder-{}.bin",
            std::process::id()
        ));
        let mut b = DiskIndexBuilder::new(2, Metric::L2).unwrap();
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
        let path = std::env::temp_dir().join(format!(
            "vanedb_test_mmap_roundtrip-{}.bin",
            std::process::id()
        ));

        let mut b = DiskIndexBuilder::new(3, Metric::L2).unwrap();
        b.add(10, &[0.0, 0.0, 0.0]).unwrap();
        b.add(20, &[1.0, 0.0, 0.0]).unwrap();
        b.add(30, &[10.0, 10.0, 10.0]).unwrap();
        b.save(&path).unwrap();

        // SAFETY: this test does not modify the file while it is mapped.
        let store = unsafe { DiskIndex::open(&path) }.unwrap();
        assert_eq!(store.size(), 3);
        assert_eq!(store.dimension(), 3);
        assert!(store.contains(10));
        assert!(!store.contains(99));

        // Get (zero-copy)
        assert_eq!(store.get(10).unwrap().as_ref(), [0.0, 0.0, 0.0]);
        assert_eq!(store.get(20).unwrap().as_ref(), [1.0, 0.0, 0.0]);

        // Search
        let results = store.search(&[0.0, 0.1, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 10);
        assert_eq!(results[1].id, 20);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_bad_file() {
        let path =
            std::env::temp_dir().join(format!("vanedb_test_mmap_bad-{}.bin", std::process::id()));
        std::fs::write(&path, b"garbage").unwrap();
        // SAFETY: this test does not modify the file while it is mapped.
        assert!(unsafe { DiskIndex::open(&path) }.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_rejects_truncated_file() {
        let path =
            std::env::temp_dir().join(format!("vanedb_test_mmap_trunc-{}.bin", std::process::id()));
        let mut data = Vec::new();
        data.extend_from_slice(&MAGIC.to_le_bytes());
        data.extend_from_slice(&VERSION.to_le_bytes());
        data.extend_from_slice(&(3u64).to_le_bytes());
        data.extend_from_slice(&(1000u64).to_le_bytes());
        data.extend_from_slice(&(0u32).to_le_bytes());
        data.extend_from_slice(&(0u32).to_le_bytes());
        std::fs::write(&path, &data).unwrap();
        // SAFETY: this test does not modify the file while it is mapped.
        assert!(unsafe { DiskIndex::open(&path) }.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn search_wrong_dimension() {
        let path =
            std::env::temp_dir().join(format!("vanedb_test_mmap_dim-{}.bin", std::process::id()));
        let mut b = DiskIndexBuilder::new(3, Metric::L2).unwrap();
        b.add(1, &[1.0, 2.0, 3.0]).unwrap();
        b.save(&path).unwrap();

        // SAFETY: this test does not modify the file while it is mapped.
        let store = unsafe { DiskIndex::open(&path) }.unwrap();
        assert!(store.search(&[1.0, 2.0], 1).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
