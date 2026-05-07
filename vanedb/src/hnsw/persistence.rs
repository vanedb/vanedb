use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use super::{HnswIndex, Inner, MAX_LEVEL};
use crate::distance::{distance_fn, DistanceMetric};
use crate::error::{Result, VaneError};

/// On-disk magic ("HNSW" little-endian) and format version.
///
/// Mirrors the framing in vanedb-cpp src/core/hnsw_index.h. The C++ side
/// uses a different MAGIC (legacy "QVRD") because it needs on-disk
/// compatibility with files written before the rename. Rust never shipped
/// pre-rename, so we use a clean per-format magic.
const MAGIC: u32 = u32::from_le_bytes(*b"HNSW");
const VERSION: u32 = 1;
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

impl HnswIndex {
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let inner = self.inner.read();
        let data = HnswData {
            dim: self.dim,
            metric: metric_to_u32(self.metric),
            max_elements: self.max_elements,
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
            vectors: inner.vectors.clone(),
            ext_ids: inner.ext_ids.clone(),
            levels: inner.levels.clone(),
            neighbors: inner.neighbors.clone(),
            id_map: inner.id_map.clone(),
        };

        let payload =
            bincode::serialize(&data).map_err(|e| VaneError::Io(format!("serialize: {e}")))?;

        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        let mut f = fs::File::create(&tmp).map_err(|e| VaneError::Io(format!("create: {e}")))?;
        f.write_all(&MAGIC.to_le_bytes())
            .map_err(|e| VaneError::Io(format!("write: {e}")))?;
        f.write_all(&VERSION.to_le_bytes())
            .map_err(|e| VaneError::Io(format!("write: {e}")))?;
        f.write_all(&payload)
            .map_err(|e| VaneError::Io(format!("write: {e}")))?;
        // Durability: fsync data + metadata before rename so a crash mid-write
        // can't leave a half-written file in place. Mirrors fsync_file in
        // vanedb-cpp src/core/detail/file_utils.h.
        f.sync_all()
            .map_err(|e| VaneError::Io(format!("sync: {e}")))?;
        drop(f);

        fs::rename(&tmp, path).map_err(|e| VaneError::Io(format!("rename: {e}")))?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = fs::read(path.as_ref()).map_err(|e| VaneError::Io(format!("read: {e}")))?;
        if bytes.len() < HEADER_LEN {
            return Err(VaneError::Io("file too small for header".to_string()));
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(VaneError::Io("invalid magic".to_string()));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(VaneError::Io(format!("unsupported version: {version}")));
        }

        let data: HnswData = bincode::deserialize(&bytes[HEADER_LEN..])
            .map_err(|e| VaneError::Io(format!("deserialize: {e}")))?;

        // Validate semantic invariants. bincode catches schema mismatches but
        // never validates values, so a corrupt or hostile file could otherwise
        // claim e.g. count > max_elements, an out-of-range entry point, or
        // neighbor indices pointing past the live set. Each check below maps
        // to one in the C++ load() path (vanedb-cpp src/core/hnsw_index.h).
        let metric = u32_to_metric(data.metric)?;
        if data.dim == 0 {
            return Err(VaneError::Io("invalid dim: 0".to_string()));
        }
        if data.max_elements == 0 {
            return Err(VaneError::Io("invalid max_elements: 0".to_string()));
        }
        if data.count > data.max_elements {
            return Err(VaneError::Io(format!(
                "corrupted file: count {} exceeds max_elements {}",
                data.count, data.max_elements
            )));
        }
        if data.max_level > MAX_LEVEL {
            return Err(VaneError::Io(format!(
                "corrupted file: max_level {} exceeds bound {}",
                data.max_level, MAX_LEVEL
            )));
        }
        match data.entry_point {
            Some(ep) => {
                if ep >= data.count {
                    return Err(VaneError::Io(format!(
                        "corrupted file: entry point {ep} >= count {}",
                        data.count
                    )));
                }
                if data.max_level < 0 {
                    return Err(VaneError::Io(
                        "corrupted file: entry point set but max_level < 0".to_string(),
                    ));
                }
            }
            None => {
                if data.count > 0 {
                    return Err(VaneError::Io(
                        "corrupted file: count > 0 with no entry point".to_string(),
                    ));
                }
            }
        }
        if data.id_map.len() != data.count {
            return Err(VaneError::Io(format!(
                "corrupted file: id_map size {} != count {}",
                data.id_map.len(),
                data.count
            )));
        }
        for &iid in data.id_map.values() {
            if iid >= data.count {
                return Err(VaneError::Io(
                    "corrupted file: id_map value out of range".to_string(),
                ));
            }
        }
        if data.neighbors.len() > data.max_elements {
            return Err(VaneError::Io(
                "corrupted file: neighbors size exceeds max_elements".to_string(),
            ));
        }
        for nbs in data.neighbors.iter().take(data.count) {
            if nbs.len() > (MAX_LEVEL as usize) + 1 {
                return Err(VaneError::Io(
                    "corrupted file: too many neighbor levels".to_string(),
                ));
            }
            for layer in nbs {
                for &n in layer {
                    if n >= data.count {
                        return Err(VaneError::Io(
                            "corrupted file: neighbor index out of range".to_string(),
                        ));
                    }
                }
            }
        }
        let expected_vecs_len = data
            .max_elements
            .checked_mul(data.dim)
            .ok_or_else(|| VaneError::Io("size overflow".to_string()))?;
        if data.vectors.len() != expected_vecs_len {
            return Err(VaneError::Io(
                "corrupted file: vectors length != max_elements * dim".to_string(),
            ));
        }
        if data.ext_ids.len() != data.max_elements || data.levels.len() != data.max_elements {
            return Err(VaneError::Io(
                "corrupted file: ext_ids/levels length != max_elements".to_string(),
            ));
        }

        // Reconstitute RNG: seed from the original seed, then advance through
        // `count` get_level calls so the next add() resumes the original
        // sequence. This is the equivalent of the C++ v2 file format that
        // serializes the std::mt19937 engine state directly.
        let mut rng = StdRng::seed_from_u64(data.seed);
        for _ in 0..data.count {
            let _ = HnswIndex::get_level(&mut rng, data.mult);
        }

        Ok(HnswIndex {
            dim: data.dim,
            metric,
            dist_fn: distance_fn(metric),
            max_elements: data.max_elements,
            m: data.m,
            m_max: data.m_max,
            m_max0: data.m_max0,
            ef_construction: data.ef_construction,
            ef_search: AtomicUsize::new(data.ef_search),
            mult: data.mult,
            seed: data.seed,
            inner: RwLock::new(Inner {
                vectors: data.vectors,
                ext_ids: data.ext_ids,
                id_map: data.id_map,
                levels: data.levels,
                neighbors: data.neighbors,
                entry_point: data.entry_point,
                max_level: data.max_level,
                count: data.count,
                rng,
            }),
        })
    }
}
