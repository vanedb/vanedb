use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::AtomicUsize;

use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Deserialize;

use super::derive_mult;
use super::{ApproxIndex, Inner, MAX_LEVEL};
use crate::distance::{distance_fn, Metric};
use crate::error::{Result, VaneError};

use super::storage::ChunkedVectors;
use super::MAX_ELEMENTS;

// Legacy Rust framing. New saves use the separate VNDB v2 graph codec.
const MAGIC: u32 = u32::from_le_bytes(*b"HNSW");
// Legacy v1 stores capacity-sized arrays; legacy v2 stores count-sized arrays.
// Both remain readable, with the same field order and bincode-1 wire layout.
const VERSION: u32 = 2;
const V1: u32 = 1;
const HEADER_LEN: usize = 8;

#[derive(Deserialize)]
pub(super) struct HnswData {
    pub(super) dim: usize,
    pub(super) metric: u32,
    pub(super) max_elements: usize,
    pub(super) m: usize,
    pub(super) m_max: usize,
    pub(super) m_max0: usize,
    pub(super) ef_construction: usize,
    pub(super) ef_search: usize,
    pub(super) _mult: f64,
    /// Original seed; replayed forward `count` times on load to restore
    /// determinism for subsequent inserts. Matches the v2 RNG-state
    /// preservation in the C++ implementation, just stored as the seed
    /// rather than a serialized engine.
    pub(super) seed: u64,
    pub(super) count: usize,
    pub(super) entry_point: Option<usize>,
    pub(super) max_level: i32,
    pub(super) vectors: Vec<f32>,
    pub(super) ext_ids: Vec<u64>,
    pub(super) levels: Vec<i32>,
    pub(super) neighbors: Vec<Vec<Vec<usize>>>,
    pub(super) id_map: HashMap<u64, usize>,
}

pub(super) fn metric_to_u32(m: Metric) -> u32 {
    match m {
        Metric::L2 => 0,
        Metric::Cosine => 1,
        Metric::Dot => 2,
    }
}

/// Preserve foreign continuation data until an insertion or rebuild replaces it.
pub(crate) struct RngState {
    pub(crate) kind: u32,
    pub(crate) bytes: Vec<u8>,
    pub(crate) count: usize,
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
    /// Writes a shared VNDB v2 graph to `path`, preserving stored slots and
    /// links.
    ///
    /// The current `ef_search` is written too and comes back from
    /// [`load`](Self::load), so a tuned index can be shipped as one file rather
    /// than a file plus a configuration note. Tombstoned slots are written as
    /// well; compact first if the file should not carry them.
    ///
    /// Rust and C++ readers support this format. Older readers do not. To
    /// migrate a legacy Rust graph, load it and save to a new path.
    ///
    /// Written beside the destination and renamed in after an fsync, so an
    /// interrupted write cannot replace a good index with a partial one.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let inner = self.inner.read();
        let path = path.as_ref();
        let temp = crate::atomic_write::AtomicFile::new(path);
        let file = fs::File::create(temp.path()).map_err(|e| VaneError::from_io("create", e))?;
        let mut f = BufWriter::new(file);
        super::graph_format::write(&mut f, self, &inner)
            .map_err(|e| VaneError::from_io("write", e))?;
        f.flush().map_err(|e| VaneError::from_io("flush", e))?;
        // Durability: fsync data + metadata before rename so a crash mid-write
        // can't leave a half-written file in place. Mirrors fsync_file in
        // vanedb-cpp src/core/detail/file_utils.h.
        f.get_ref()
            .sync_all()
            .map_err(|e| VaneError::from_io("sync", e))?;
        drop(f);

        temp.commit(path)
    }

    /// Reads an index written by [`save`](Self::save).
    ///
    /// Accepts shared VNDB v2 graphs and legacy Rust v1/v2 files. Structural
    /// invariants are validated before the index is exposed. Inserting into a
    /// graph from another engine may produce different topology.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = fs::read(path.as_ref()).map_err(|e| VaneError::from_io("read", e))?;
        if bytes.len() < HEADER_LEN {
            return Err(VaneError::corrupt("file too small for header"));
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let (version, data, persisted_rng) = if magic
            == u32::from_le_bytes(*super::graph_format::MAGIC)
        {
            let (data, rng) = super::graph_format::read(&bytes)?;
            (VERSION, data, Some(rng))
        } else {
            if magic != MAGIC {
                return Err(VaneError::corrupt("invalid magic"));
            }
            let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
            if version != V1 && version != VERSION {
                return Err(VaneError::corrupt(format!(
                    "unsupported version: {version}"
                )));
            }
            // Keep bincode 1's fixed-width little-endian wire layout readable.
            let (data, _): (HnswData, usize) =
                bincode::serde::decode_from_slice(&bytes[HEADER_LEN..], bincode::config::legacy())
                    .map_err(|e| VaneError::corrupt(format!("deserialize: {e}")))?;
            (version, data, None)
        };

        // Both decoders share the same semantic validation. In particular,
        // legacy bincode checks the schema but accepts impossible counts,
        // entry points and edges unless validated here.
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
        // every array a file declares the same way, against the same 100
        // million bound (`MAX_VEC_SIZE` in `cpp/src/core/approx_index.h`).
        // That was not true when this comment was first written: the C++ cap
        // reached only the arrays read through `read_vec`, while its legacy
        // loader pre-allocated `max_elements * dimension` straight from the
        // header with overflow as its only bound.
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
        let observed_max = data
            .levels
            .iter()
            .take(data.count)
            .copied()
            .max()
            .unwrap_or(-1);
        if data.max_level != observed_max
            || data
                .entry_point
                .is_some_and(|ep| data.levels[ep] != data.max_level)
        {
            return Err(VaneError::corrupt(
                "entry point/max_level disagrees with node levels",
            ));
        }
        for (iid, nbs) in data.neighbors.iter().take(data.count).enumerate() {
            let level = data.levels[iid];
            if !(0..=MAX_LEVEL).contains(&level) || nbs.len() != level as usize + 1 {
                return Err(VaneError::corrupt(
                    "neighbor layers disagree with node level",
                ));
            }
            for (layer_index, layer) in nbs.iter().enumerate() {
                let cap = if layer_index == 0 {
                    data.m_max0
                } else {
                    data.m_max
                };
                if layer.len() > cap {
                    return Err(VaneError::corrupt("neighbor degree exceeds layer limit"));
                }
                let mut seen = HashSet::with_capacity(layer.len());
                for &n in layer {
                    if n >= data.count
                        || n == iid
                        || !seen.insert(n)
                        || data.levels[n] < layer_index as i32
                    {
                        return Err(VaneError::corrupt("invalid or duplicate neighbor in layer"));
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
        // Storage grows on demand. Retain only stored slots, dropping unused
        // capacity from legacy v1 arrays.
        let vectors = ChunkedVectors::from_flat(data.dim, &data.vectors[..live_vectors_len]);
        let mut ext_ids = data.ext_ids;
        ext_ids.truncate(data.count);
        let mut levels = data.levels;
        levels.truncate(data.count);
        let mut neighbors = data.neighbors;
        neighbors.truncate(data.count);

        // A slot is live exactly when id_map maps its ext_id back to it.
        // Legacy files encode liveness through this map; the VNDB reader
        // rebuilds the same map from explicit node flags.
        let deleted_flags: Vec<bool> = (0..data.count)
            .map(|iid| data.id_map.get(&ext_ids[iid]) != Some(&iid))
            .collect();
        let live = data.count - deleted_flags.iter().filter(|d| **d).count();

        // Resume Rust's level sequence from seed and stored count. For a
        // foreign generator this is the fallback used on the next insertion;
        // unmodified saves retain the original continuation metadata.
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
                persisted_rng,
            }),
        })
    }
}

#[cfg(test)]
mod legacy_fixtures {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn fixed_legacy_files_preserve_graph_identity_and_mutability() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/legacy_graph");
        for (name, metric, deleted) in [
            ("v1_l2", Metric::L2, false),
            ("v1_cosine", Metric::Cosine, false),
            ("v1_dot", Metric::Dot, false),
            ("v2_l2", Metric::L2, false),
            ("v2_cosine", Metric::Cosine, false),
            ("v2_dot", Metric::Dot, false),
            ("v2_deleted_id_reuse", Metric::Cosine, true),
        ] {
            let index = ApproxIndex::load(fixtures.join(format!("{name}.hnsw"))).unwrap();
            assert_eq!(index.dim, 2, "{name}");
            assert_eq!(index.metric, metric, "{name}");
            assert_eq!(index.max_elements, 4);
            assert_eq!((index.m, index.ef_construction, index.seed), (2, 16, 42));
            assert_eq!(index.ef_search.load(Ordering::Relaxed), 16);
            {
                let inner = index.inner.read();
                assert_eq!(inner.count, 3);
                assert_eq!((inner.entry_point, inner.max_level), (Some(0), 1));
                assert_eq!(inner.levels, [1, 0, 1]);
                assert_eq!(inner.deleted, [false, deleted, false]);
                assert_eq!(
                    inner.neighbors,
                    [
                        vec![vec![1, 2], vec![2]],
                        vec![vec![0, 2]],
                        vec![vec![0, 1], vec![0]],
                    ]
                );
                assert_eq!(inner.vectors.to_flat(3), [1.0, 0.0, 0.0, 1.0, 0.8, 0.2]);
            }
            let results = index.search(&[1.0, 0.0], 3).unwrap();
            let expected = if deleted {
                vec![101, u64::MAX]
            } else {
                vec![101, u64::MAX, 202]
            };
            assert_eq!(
                results.iter().map(|r| r.id).collect::<Vec<_>>(),
                expected,
                "{name}"
            );
            assert_eq!(
                results[0].distance,
                if metric == Metric::Dot { -1.0 } else { 0.0 }
            );
            assert_eq!(index.contains(202), !deleted);
            assert_eq!(index.get_vector(101).unwrap(), [1.0, 0.0]);
            index.add(303, &[0.25, 0.75]).unwrap();
            let path = std::env::temp_dir()
                .join(format!("vanedb-legacy-{}-{name}.hnsw", std::process::id()));
            index.save(&path).unwrap();
            let reloaded = ApproxIndex::load(&path).unwrap();
            fs::remove_file(path).unwrap();
            assert_eq!(
                reloaded.inner.read().neighbors,
                index.inner.read().neighbors
            );
            assert_eq!(reloaded.inner.read().levels, index.inner.read().levels);
            assert_eq!(reloaded.get_vector(303).unwrap(), [0.25, 0.75]);
        }
    }
}
