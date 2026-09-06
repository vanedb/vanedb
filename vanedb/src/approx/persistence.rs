use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use super::derive_mult;
use super::{ApproxIndex, Inner, MAX_LEVEL};
use crate::distance::{distance_fn, Metric};
use crate::error::{Result, VaneError};

use super::storage::ChunkedVectors;
use super::MAX_ELEMENTS;

/// On-disk magic ("HNSW" little-endian) and format version.
///
/// Mirrors the framing in vanedb-cpp src/core/index.h. The C++ side
/// uses a different MAGIC (legacy "QVRD") because it needs on-disk
/// compatibility with files written before the rename. Rust never shipped
/// pre-rename, so we use a clean per-format magic.
const MAGIC: u32 = u32::from_le_bytes(*b"HNSW");
/// v1 stored the full pre-allocated arrays (`max_elements` entries even when
/// only `count` were inserted). v2 stores only the `count` live entries and
/// re-expands to `max_elements` on load (issue #18). v1 files remain loadable.
const VERSION: u32 = 2;
const V1: u32 = 1;
const HEADER_LEN: usize = 8;

#[derive(Serialize, Deserialize)]
struct HnswData {
    dim: usize,
    metric: u32,
    max_elements: usize,
    m: usize,
    m_max: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    mult: f64,
    /// Original seed; replayed forward `count` times on load to restore
    /// determinism for subsequent inserts. Matches the v2 RNG-state
    /// preservation in the C++ implementation, just stored as the seed
    /// rather than a serialized engine.
    seed: u64,
    count: usize,
    entry_point: Option<usize>,
    max_level: i32,
    vectors: Vec<f32>,
    ext_ids: Vec<u64>,
    levels: Vec<i32>,
    neighbors: Vec<Vec<Vec<usize>>>,
    id_map: HashMap<u64, usize>,
}

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

impl ApproxIndex {
    /// Writes the index to `path`.
    ///
    /// Written beside the destination and renamed in after an fsync, so an
    /// interrupted write cannot replace a good index with a partial one.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let inner = self.inner.read();
        let data = HnswData {
            dim: self.dim,
            metric: metric_to_u32(self.metric),
            // A capacity hint that has been grown past would describe a file
            // that cannot be reloaded, so record what was actually written.
            max_elements: self.max_elements.max(inner.count),
            m: self.m,
            m_max: self.m_max,
            m_max0: self.m_max0,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search.load(Ordering::Relaxed),
            mult: self.mult,
            seed: self.seed,
            count: inner.count,
            entry_point: inner.entry_point,
            max_level: inner.max_level,
            // v2: persist only the `count` live entries, not the full
            // pre-allocated capacity; load() re-expands to max_elements.
            vectors: inner.vectors.to_flat(inner.count),
            ext_ids: inner.ext_ids[..inner.count].to_vec(),
            levels: inner.levels[..inner.count].to_vec(),
            neighbors: inner.neighbors[..inner.count].to_vec(),
            id_map: inner.id_map.clone(),
        };

        let payload = bincode::serde::encode_to_vec(&data, bincode::config::legacy())
            .map_err(|e| VaneError::from_io("serialize", std::io::Error::other(e)))?;

        let path = path.as_ref();
        let temp = crate::atomic_write::AtomicFile::new(path);
        let mut f = fs::File::create(temp.path()).map_err(|e| VaneError::from_io("create", e))?;
        f.write_all(&MAGIC.to_le_bytes())
            .map_err(|e| VaneError::from_io("write", e))?;
        f.write_all(&VERSION.to_le_bytes())
            .map_err(|e| VaneError::from_io("write", e))?;
        f.write_all(&payload)
            .map_err(|e| VaneError::from_io("write", e))?;
        // Durability: fsync data + metadata before rename so a crash mid-write
        // can't leave a half-written file in place. Mirrors fsync_file in
        // vanedb-cpp src/core/detail/file_utils.h.
        f.sync_all().map_err(|e| VaneError::from_io("sync", e))?;
        drop(f);

        temp.commit(path)
    }

    /// Reads an index written by [`save`](Self::save).
    ///
    /// Files from earlier released versions stay loadable. Structural
    /// invariants are checked on the way in, so a corrupt file is rejected
    /// rather than producing wrong search results.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = fs::read(path.as_ref()).map_err(|e| VaneError::from_io("read", e))?;
        if bytes.len() < HEADER_LEN {
            return Err(VaneError::corrupt("file too small for header"));
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(VaneError::corrupt("invalid magic"));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != V1 && version != VERSION {
            return Err(VaneError::corrupt(format!(
                "unsupported version: {version}"
            )));
        }

        // config::legacy() is bincode 1's wire format (fixint, little-endian),
        // so files written before the bincode 2 migration load unchanged.
        let (data, _): (HnswData, usize) =
            bincode::serde::decode_from_slice(&bytes[HEADER_LEN..], bincode::config::legacy())
                .map_err(|e| VaneError::corrupt(format!("deserialize: {e}")))?;

        // Validate semantic invariants. bincode catches schema mismatches but
        // never validates values, so a corrupt or hostile file could otherwise
        // claim e.g. count > max_elements, an out-of-range entry point, or
        // neighbor indices pointing past the live set. Each check below maps
        // to one in the C++ load() path (vanedb-cpp src/core/index.h).
        let metric = u32_to_metric(data.metric)?;
        if data.dim == 0 {
            return Err(VaneError::corrupt("invalid dim: 0"));
        }
        if data.dim.checked_mul(std::mem::size_of::<f32>()).is_none() {
            return Err(VaneError::corrupt(format!(
                "corrupted file: dim {} overflows when sized in bytes",
                data.dim
            )));
        }
        if data.max_elements == 0 {
            return Err(VaneError::corrupt("invalid max_elements: 0"));
        }
        // A file declares max_elements; load re-expands every array to it. Cap
        // it so a few hundred bytes cannot request terabytes. vanedb-cpp caps
        // every deserialized array the same way (MAX_VEC_SIZE in
        // src/core/index.h).
        if data.max_elements > MAX_ELEMENTS {
            return Err(VaneError::corrupt(format!(
                "corrupted file: max_elements {} exceeds the {MAX_ELEMENTS} limit",
                data.max_elements
            )));
        }
        // C++ validates these on load; Rust checked them only at build time,
        // so a hostile file could set m = 0 and make the level maths degenerate.
        if data.m < 2 {
            return Err(VaneError::corrupt(format!(
                "corrupted file: m {} is below 2",
                data.m
            )));
        }
        if data.ef_construction == 0 {
            return Err(VaneError::corrupt("corrupted file: ef_construction is 0"));
        }
        let m_max0_expected = data.m.checked_mul(2).ok_or_else(|| {
            VaneError::corrupt(format!(
                "corrupted file: m {} overflows when doubled",
                data.m
            ))
        })?;
        if data.m_max0 != m_max0_expected || data.m_max != data.m {
            return Err(VaneError::corrupt(format!(
                "corrupted file: m_max {} / m_max0 {} disagree with m {}",
                data.m_max, data.m_max0, data.m
            )));
        }
        // max_elements is a capacity *hint*, and growing past it is supported
        // (#90), so it does not bound count in v2. v1 files pre-allocated
        // their arrays to max_elements, so there the two really are tied and
        // the array-length check below enforces it.
        if version == V1 && data.count > data.max_elements {
            return Err(VaneError::corrupt(format!(
                "corrupted file: count {} exceeds max_elements {}",
                data.count, data.max_elements
            )));
        }
        if data.count > MAX_ELEMENTS {
            return Err(VaneError::corrupt(format!(
                "corrupted file: count {} exceeds the {MAX_ELEMENTS} element cap",
                data.count
            )));
        }
        if data.max_level > MAX_LEVEL {
            return Err(VaneError::corrupt(format!(
                "corrupted file: max_level {} exceeds bound {}",
                data.max_level, MAX_LEVEL
            )));
        }
        match data.entry_point {
            Some(ep) => {
                if ep >= data.count {
                    return Err(VaneError::corrupt(format!(
                        "corrupted file: entry point {ep} >= count {}",
                        data.count
                    )));
                }
                if data.max_level < 0 {
                    return Err(VaneError::corrupt(
                        "corrupted file: entry point set but max_level < 0",
                    ));
                }
            }
            None => {
                if data.count > 0 {
                    return Err(VaneError::corrupt(
                        "corrupted file: count > 0 with no entry point",
                    ));
                }
            }
        }
        // With tombstones id_map holds only the live ids, so it may be
        // smaller than count. More than count is still corrupt: every entry
        // must name a distinct slot.
        if data.id_map.len() > data.count {
            return Err(VaneError::corrupt(format!(
                "corrupted file: id_map size {} exceeds count {}",
                data.id_map.len(),
                data.count
            )));
        }
        // Every live slot must be reachable by its own external id. Combined
        // with the length check above this forces a bijection, which rejects
        // key/value mismatches, duplicate external ids, duplicated internal
        // ids, and missing entries in one pass. The previous length-and-range
        // check accepted `ext_ids[0] = 10` alongside `id_map = {20: 0}`, so a
        // lookup of 20 returned slot 0's vector under the wrong identity (#42).
        // Tombstones make this one-directional: a deleted slot keeps its
        // ext_id but is absent from id_map, and its id may since have been
        // reused by a later slot. So every id_map entry must point at a slot
        // carrying that id, and a slot absent from id_map is simply deleted.
        // Length first: the id_map loop below indexes ext_ids, so a file
        // claiming more elements than it carries would panic on the untrusted
        // path rather than returning an error.
        let stored = if version == V1 {
            data.max_elements
        } else {
            data.count
        };
        if data.ext_ids.len() != stored || data.levels.len() != stored {
            return Err(VaneError::corrupt(
                "corrupted file: ext_ids/levels length != expected",
            ));
        }

        for (&ext_id, &mapped) in &data.id_map {
            if mapped >= data.count {
                return Err(VaneError::corrupt(format!(
                    "corrupted file: id_map[{ext_id}] is {mapped}, beyond count {}",
                    data.count
                )));
            }
            if data.ext_ids[mapped] != ext_id {
                return Err(VaneError::corrupt(format!(
                    "corrupted file: id_map[{ext_id}] is {mapped}, whose ext_id is {}",
                    data.ext_ids[mapped]
                )));
            }
        }
        if data.neighbors.len() != stored {
            return Err(VaneError::corrupt(format!(
                "corrupted file: neighbors length {} != expected {stored}",
                data.neighbors.len()
            )));
        }
        for nbs in data.neighbors.iter().take(data.count) {
            if nbs.len() > (MAX_LEVEL as usize) + 1 {
                return Err(VaneError::corrupt(
                    "corrupted file: too many neighbor levels",
                ));
            }
            for layer in nbs {
                for &n in layer {
                    if n >= data.count {
                        return Err(VaneError::corrupt(
                            "corrupted file: neighbor index out of range",
                        ));
                    }
                }
            }
        }
        data.max_elements.checked_mul(data.dim).ok_or_else(|| {
            VaneError::corrupt(format!(
                "corrupted file: max_elements {} * dim {} overflows",
                data.max_elements, data.dim
            ))
        })?;
        // `stored` is max_elements for v1 but `count` for v2, and v2 does not
        // bound count by max_elements — so this product is over two values the
        // file controls. At 8 * 2^61 it wraps to exactly 0, and an empty
        // vector array then satisfies the length check below.
        let stored_len = stored.checked_mul(data.dim).ok_or_else(|| {
            VaneError::corrupt(format!(
                "corrupted file: stored {} * dim {} overflows",
                stored, data.dim
            ))
        })?;
        if data.vectors.len() != stored_len {
            return Err(VaneError::corrupt(
                "corrupted file: vectors length != expected * dim",
            ));
        }
        let live_vectors_len = data.count.checked_mul(data.dim).ok_or_else(|| {
            VaneError::corrupt(format!(
                "corrupted file: count {} * dim {} overflows",
                data.count, data.dim
            ))
        })?;
        if data.vectors[..live_vectors_len]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(VaneError::corrupt(
                "corrupted file: vector values must be finite",
            ));
        }
        // Re-expand to the pre-allocated capacity layout Inner expects. For
        // v1 the arrays are already full-length, so these are no-ops.
        //
        // Reserve fallibly first: `resize` aborts the process on allocation
        // failure, and this is the path that reads untrusted files. The cap
        // above bounds the request; this turns a genuine OOM into an error
        // rather than killing the host application (#89).
        // Storage grows on demand, so only the live vectors are rebuilt.
        let vectors = ChunkedVectors::from_flat(data.dim, &data.vectors[..live_vectors_len]);
        let mut ext_ids = data.ext_ids;
        ext_ids.truncate(data.count);
        let mut levels = data.levels;
        levels.truncate(data.count);
        let mut neighbors = data.neighbors;
        neighbors.truncate(data.count);

        // A slot is live exactly when id_map maps its ext_id back to it. That
        // is the same relation the check above enforces, read the other way,
        // so tombstones need no field in the file format and v1/v2 files
        // (which have no deletions) load with every slot live.
        let deleted_flags: Vec<bool> = (0..data.count)
            .map(|iid| data.id_map.get(&ext_ids[iid]) != Some(&iid))
            .collect();
        let live = data.count - deleted_flags.iter().filter(|d| **d).count();

        // Reconstitute RNG: seed from the original seed, then advance through
        // `count` get_level calls so the next add() resumes the original
        // sequence. This is the equivalent of the C++ v2 file format that
        // serializes the std::mt19937 engine state directly.
        let mut rng = StdRng::seed_from_u64(data.seed);
        for _ in 0..data.count {
            let _ = ApproxIndex::get_level(&mut rng, derive_mult(data.m));
        }

        Ok(ApproxIndex {
            dim: data.dim,
            metric,
            dist_fn: distance_fn(metric),
            max_elements: data.max_elements,
            m: data.m,
            m_max: data.m_max,
            m_max0: data.m_max0,
            ef_construction: data.ef_construction,
            ef_search: AtomicUsize::new(data.ef_search),
            // Recomputed, not trusted: mult is fully derived from m, and a
            // negative value from a crafted file made get_level return a
            // negative level, which `0..=level as usize` wrapped into a
            // ~2^64 range and aborted the process on the next add().
            mult: derive_mult(data.m),
            seed: data.seed,
            inner: RwLock::new(Inner {
                deleted: deleted_flags,
                live,
                vectors,
                ext_ids,
                id_map: data.id_map,
                levels,
                neighbors,
                entry_point: data.entry_point,
                max_level: data.max_level,
                count: data.count,
                rng,
            }),
        })
    }
}
