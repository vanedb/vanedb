//! Corruption and validation tests for HNSW persistence and mmap files.
//!
//! Mirrors the corruption/validation suite from vanedb-cpp PR #5
//! (tests/test_hnsw_index.cpp + tests/test_mmap_vector_store.cpp).

use std::fs;
use std::io::Write;

use vanedb::{ApproxIndex, Metric, VaneError};

#[cfg(feature = "disk")]
use vanedb::{DiskIndex, DiskIndexBuilder};

const HNSW_MAGIC: u32 = u32::from_le_bytes(*b"HNSW");
/// Must match `disk::MAGIC`. Written as bytes rather than a hex literal so it
/// cannot be transcribed byte-reversed — a wrong magic makes every header test
/// below reject on magic without reaching the guard it names.
const DISK_MAGIC: u32 = u32::from_le_bytes(*b"VNDB");
const HNSW_VERSION: u32 = 2;

/// Field-order mirror of the private `HnswData` struct in
/// `src/approx/persistence.rs` (bincode encodes by field order, so this
/// serializes identically). Used to hand-craft v1/v2 payloads.
#[derive(serde::Serialize)]
struct HnswDataMirror {
    dim: usize,
    metric: u32,
    max_elements: usize,
    m: usize,
    m_max: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    mult: f64,
    seed: u64,
    count: usize,
    entry_point: Option<usize>,
    max_level: i32,
    vectors: Vec<f32>,
    ext_ids: Vec<u64>,
    levels: Vec<i32>,
    neighbors: Vec<Vec<Vec<usize>>>,
    id_map: std::collections::HashMap<u64, usize>,
}

/// A consistent 2-of-4-slots index in the legacy v1 layout: arrays span the
/// full pre-allocated capacity, not just the inserted count.
fn v1_full_capacity_payload() -> HnswDataMirror {
    HnswDataMirror {
        dim: 2,
        metric: 0, // L2
        max_elements: 4,
        m: 2,
        m_max: 2,
        m_max0: 4,
        ef_construction: 10,
        ef_search: 10,
        mult: 1.0,
        seed: 7,
        count: 2,
        entry_point: Some(0),
        max_level: 0,
        vectors: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ext_ids: vec![10, 20, 0, 0],
        levels: vec![0; 4],
        neighbors: vec![vec![vec![1]], vec![vec![0]], vec![], vec![]],
        id_map: std::collections::HashMap::from([(10, 0), (20, 1)]),
    }
}

fn hnsw_file_bytes(version: u32, data: &HnswDataMirror) -> Vec<u8> {
    let mut bytes = HNSW_MAGIC.to_le_bytes().to_vec();
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(
        &bincode::serde::encode_to_vec(data, bincode::config::legacy()).unwrap(),
    );
    bytes
}

#[test]
fn hnsw_load_accepts_v1_full_capacity_files() {
    let bytes = hnsw_file_bytes(1, &v1_full_capacity_payload());
    let p = write_tmp("v1_compat", &bytes);
    let idx = ApproxIndex::load(&p).unwrap();
    assert_eq!(idx.size(), 2);
    assert_eq!(idx.capacity(), 4);
    assert_eq!(idx.get_vector(10).unwrap(), vec![1.0, 0.0]);
    let results = idx.search(&[1.0, 0.1], 1).unwrap();
    assert_eq!(results[0].id, 10);
    // Spare capacity from the v1 file must remain usable.
    idx.add(30, &[0.5, 0.5]).unwrap();
    assert_eq!(idx.size(), 3);
    let _ = fs::remove_file(&p);
}

#[test]
fn hnsw_load_rejects_non_finite_stored_vectors() {
    let mut data = v1_full_capacity_payload();
    data.vectors[0] = f32::NAN;
    let bytes = hnsw_file_bytes(1, &data);
    let p = write_tmp("non_finite_vector", &bytes);
    let err = match ApproxIndex::load(&p) {
        Ok(_) => panic!("load should have failed"),
        Err(error) => error,
    };
    assert!(format!("{err}").contains("finite"), "got: {err}");
    let _ = fs::remove_file(&p);
}

#[test]
fn hnsw_load_rejects_v2_with_capacity_sized_arrays() {
    // The same full-capacity arrays are NOT valid under v2, which stores
    // exactly `count` entries per array.
    let bytes = hnsw_file_bytes(2, &v1_full_capacity_payload());
    let p = write_tmp("v2_full_arrays", &bytes);
    let err = match ApproxIndex::load(&p) {
        Ok(_) => panic!("load should have failed"),
        Err(e) => e,
    };
    assert!(format!("{err}").contains("length"), "got: {err}");
    let _ = fs::remove_file(&p);
}

#[cfg(feature = "disk")]
#[test]
fn mmap_open_rejects_non_finite_stored_vectors() {
    let path = std::env::temp_dir().join("vanedb_mmap_non_finite.bin");
    let mut builder = DiskIndexBuilder::new(2, Metric::L2).unwrap();
    builder.add(1, &[0.0, 0.0]).unwrap();
    builder.save(&path).unwrap();

    let mut bytes = fs::read(&path).unwrap();
    let vector_offset = 32 + 8;
    bytes[vector_offset..vector_offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    fs::write(&path, bytes).unwrap();

    // SAFETY: this test does not modify the file while it is mapped.
    let err = match unsafe { DiskIndex::open(&path) } {
        Ok(_) => panic!("open should have failed"),
        Err(error) => error,
    };
    assert!(format!("{err}").contains("finite"), "got: {err}");
    let _ = fs::remove_file(&path);
}

/// Build a minimal valid HNSW file on disk, then return the bytes so tests can
/// mutate specific fields and exercise validation paths in `load`. The `tag`
/// scopes the temp-file path so parallel tests don't collide on the same name.
fn valid_hnsw_bytes(tag: &str) -> Vec<u8> {
    let path = std::env::temp_dir().join(format!("vanedb_corruption_seed_{tag}.bin"));
    let idx = ApproxIndex::builder(4, Metric::L2)
        .capacity(8)
        .seed(7)
        .build()
        .unwrap();
    idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
    idx.add(3, &[0.0, 0.0, 1.0, 0.0]).unwrap();
    idx.save(&path).unwrap();
    let bytes = fs::read(&path).unwrap();
    let _ = fs::remove_file(&path);
    bytes
}

fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("vanedb_corruption_{name}.bin"));
    let mut f = fs::File::create(&p).unwrap();
    f.write_all(bytes).unwrap();
    p
}

#[test]
fn hnsw_load_rejects_invalid_magic() {
    let mut bytes = valid_hnsw_bytes("bad_magic");
    bytes[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
    let p = write_tmp("bad_magic", &bytes);
    let err = match ApproxIndex::load(&p) {
        Ok(_) => panic!("load should have failed"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, VaneError::Corrupt { detail } if detail.contains("magic")),
        "got: {err:?}"
    );
    let _ = fs::remove_file(&p);
}

#[test]
fn hnsw_load_rejects_unsupported_version() {
    let mut bytes = valid_hnsw_bytes("bad_version");
    bytes[4..8].copy_from_slice(&999u32.to_le_bytes());
    let p = write_tmp("bad_version", &bytes);
    let err = match ApproxIndex::load(&p) {
        Ok(_) => panic!("load should have failed"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, VaneError::Corrupt { detail } if detail.contains("version")),
        "got: {err:?}"
    );
    let _ = fs::remove_file(&p);
}

#[test]
fn hnsw_load_rejects_truncated_header() {
    let p = write_tmp("trunc_header", b"HNS"); // 3 bytes — shorter than 8-byte header
    assert!(matches!(
        ApproxIndex::load(&p),
        Err(VaneError::Corrupt { .. })
    ));
    let _ = fs::remove_file(&p);
}

#[test]
fn hnsw_load_rejects_garbage_payload() {
    // Valid magic+version but bincode payload is junk.
    let mut bytes = HNSW_MAGIC.to_le_bytes().to_vec();
    bytes.extend_from_slice(&HNSW_VERSION.to_le_bytes());
    bytes.extend_from_slice(&[0xFF; 32]);
    let p = write_tmp("garbage_payload", &bytes);
    assert!(matches!(
        ApproxIndex::load(&p),
        Err(VaneError::Corrupt { .. })
    ));
    let _ = fs::remove_file(&p);
}

#[test]
fn hnsw_save_load_preserves_all_metrics() {
    for &metric in &[Metric::L2, Metric::Cosine, Metric::Dot] {
        let path = std::env::temp_dir().join(format!("vanedb_metric_{metric:?}.bin"));
        let idx = ApproxIndex::builder(3, metric).capacity(4).build().unwrap();
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        idx.save(&path).unwrap();
        let loaded = ApproxIndex::load(&path).unwrap();
        assert_eq!(loaded.metric(), metric);
        let _ = fs::remove_file(&path);
    }
}

#[test]
fn hnsw_save_load_preserves_rng_determinism() {
    // Critical regression test for the post-load RNG state. Before this port,
    // `load()` reseeded the RNG with `seed_from_u64(count as u64)` instead of
    // the original seed, so subsequent inserts diverged from a never-saved
    // index using the same builder seed. Now `load()` replays `count`
    // get_level calls so the next insert sees the same RNG state.
    let path = std::env::temp_dir().join("vanedb_rng_determinism.bin");

    // Reference: build, insert 5, then insert 5 more, never saving.
    let reference = ApproxIndex::builder(4, Metric::L2)
        .capacity(20)
        .seed(123)
        .build()
        .unwrap();
    for i in 0..10u64 {
        reference.add(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
    }

    // Round-trip: build, insert 5, save, load, insert 5 more.
    let saved = ApproxIndex::builder(4, Metric::L2)
        .capacity(20)
        .seed(123)
        .build()
        .unwrap();
    for i in 0..5u64 {
        saved.add(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
    }
    saved.save(&path).unwrap();

    let loaded = ApproxIndex::load(&path).unwrap();
    for i in 5..10u64 {
        loaded.add(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
    }

    // Search results must match: same seed + same insertion order should yield
    // identical graph topology, hence identical search results.
    let q = [3.5, 0.0, 0.0, 0.0];
    let r_ref = reference.search(&q, 5).unwrap();
    let r_loaded = loaded.search(&q, 5).unwrap();
    assert_eq!(r_ref.len(), r_loaded.len());
    for (a, b) in r_ref.iter().zip(r_loaded.iter()) {
        assert_eq!(a.id, b.id, "RNG state drift after save/load");
    }

    let _ = fs::remove_file(&path);
}

// ---- mmap corruption tests ----

#[cfg(feature = "disk")]
#[test]
fn mmap_load_rejects_nonzero_reserved_header_bytes() {
    let path = std::env::temp_dir().join(format!(
        "vanedb_reserved_header_{}.vndb",
        std::process::id()
    ));
    let mut bytes = disk_file_bytes(DISK_MAGIC, 1, 2, &[7], &[1.0, 2.0]);
    for reserved in [0_u32, 1, 0x100, 0x1_0000, 0x8000_0000, 0] {
        bytes[28..32].copy_from_slice(&reserved.to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        // SAFETY: this test file is unchanged until this iteration's map drops.
        let result = unsafe { DiskIndex::open(&path) };
        if reserved == 0 {
            assert_eq!(result.unwrap().get(7).unwrap().as_ref(), [1.0, 2.0]);
        } else {
            let error = result.unwrap_err();
            assert!(matches!(error, VaneError::Corrupt { .. }));
            assert!(error.to_string().contains("reserved"), "{error}");
        }
    }
    fs::remove_file(path).unwrap();
}

#[cfg(feature = "disk")]
#[test]
fn mmap_load_rejects_unsupported_version() {
    let path = std::env::temp_dir().join("vanedb_mmap_bad_version.bin");
    let mut data = Vec::new();
    data.extend_from_slice(&DISK_MAGIC.to_le_bytes());
    data.extend_from_slice(&999u32.to_le_bytes()); // unsupported
    data.extend_from_slice(&3u64.to_le_bytes()); // dim
    data.extend_from_slice(&0u64.to_le_bytes()); // num_vectors
    data.extend_from_slice(&0u32.to_le_bytes()); // metric
    data.extend_from_slice(&0u32.to_le_bytes()); // reserved
    fs::write(&path, &data).unwrap();
    assert!(matches!(
        // SAFETY: this test does not modify the file while it is mapped.
        unsafe { DiskIndex::open(&path) },
        Err(VaneError::Corrupt { .. })
    ));
    let _ = fs::remove_file(&path);
}

#[cfg(feature = "disk")]
#[test]
fn mmap_load_rejects_zero_dim_with_vectors() {
    // Self-consistent at dim = 0: 2 ids and no vector bytes is exactly the
    // declared length, so the truncation check passes and only the explicit
    // zero-dim guard can reject this.
    let bytes = disk_file_bytes(DISK_MAGIC, 1, 0, &[1, 2], &[]);
    let p = write_tmp("mmap_zero_dim", &bytes);
    assert!(
        // SAFETY: this test does not modify the file while it is mapped.
        matches!(
            unsafe { DiskIndex::open(&p) },
            Err(VaneError::Corrupt { .. })
        ),
        "dim = 0 with vectors present must be rejected"
    );
    let _ = fs::remove_file(&p);
}

#[cfg(feature = "disk")]
#[test]
fn mmap_load_rejects_truncated_data() {
    // Header claims 1000 vectors but file ends after the header.
    let path = std::env::temp_dir().join("vanedb_mmap_truncated.bin");
    let mut data = Vec::new();
    data.extend_from_slice(&DISK_MAGIC.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&3u64.to_le_bytes());
    data.extend_from_slice(&1000u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    fs::write(&path, &data).unwrap();
    assert!(matches!(
        // SAFETY: this test does not modify the file while it is mapped.
        unsafe { DiskIndex::open(&path) },
        Err(VaneError::Corrupt { .. })
    ));
    let _ = fs::remove_file(&path);
}

#[cfg(feature = "disk")]
#[test]
fn mmap_load_rejects_a_header_that_understates_the_payload() {
    // The mirror of `mmap_load_rejects_truncated_data`, and the case that was
    // missing. `expected` is derived FROM the header, so a header that lies in
    // the direction that makes the file look larger than declared moves the
    // goalpost instead of tripping a one-sided `len() < expected` check.
    //
    // One bit, in the low byte of `dim`: 3 becomes 2. The payload is then read
    // at the wrong stride and `get` returns a vector that straddles two stored
    // records — a value that was never written by anyone.
    let mut builder = DiskIndexBuilder::new(3, Metric::L2).unwrap();
    builder.add(11, &[1.0, 2.0, 3.0]).unwrap();
    builder.add(22, &[4.0, 5.0, 6.0]).unwrap();
    let good = std::env::temp_dir().join(format!(
        "vanedb_mmap_understated_good_{}.bin",
        std::process::id()
    ));
    builder.save(&good).unwrap();

    let mut bytes = fs::read(&good).unwrap();
    bytes[8] ^= 0x01;
    let bad = write_tmp("mmap_understated_bad", &bytes);

    // SAFETY: this test owns both files and does not modify them while mapped.
    let opened = unsafe { DiskIndex::open(&bad) };
    assert!(
        matches!(opened, Err(VaneError::Corrupt { .. })),
        "a header whose declared geometry is shorter than the file must be \
         rejected, not reinterpreted at the wrong stride: {:?}",
        opened.map(|i| (i.dimension(), i.size()))
    );

    let _ = fs::remove_file(&good);
    let _ = fs::remove_file(&bad);
}

#[cfg(feature = "disk")]
#[test]
fn mmap_load_rejects_every_single_bit_flip_in_the_geometry_fields() {
    // `dim` (offsets 8..16) and `num_vectors` (16..24) are the two fields the
    // payload length is computed from, so every flip in them makes the header
    // disagree with the file. None may be accepted: an accepted flip here is
    // silent misinterpretation, not a smaller index.
    //
    // The other header fields are covered elsewhere — magic and version have
    // their own tests, the reserved word is checked for zero, and `metric`
    // yields a structurally valid file, so no length check can reject it.
    let mut builder = DiskIndexBuilder::new(4, Metric::L2).unwrap();
    for id in 0..5u64 {
        let f = id as f32;
        builder.add(id, &[f, f + 1.0, f + 2.0, f + 3.0]).unwrap();
    }
    let good = std::env::temp_dir().join(format!(
        "vanedb_mmap_geometry_good_{}.bin",
        std::process::id()
    ));
    builder.save(&good).unwrap();
    let original = fs::read(&good).unwrap();

    let mut accepted = Vec::new();
    for byte in 8..24usize {
        for bit in 0..8u32 {
            let mut bytes = original.clone();
            bytes[byte] ^= 1 << bit;
            let path = write_tmp(&format!("mmap_geometry_{byte}_{bit}"), &bytes);
            // SAFETY: this test owns the file and does not modify it while mapped.
            if let Ok(index) = unsafe { DiskIndex::open(&path) } {
                accepted.push((byte, bit, index.dimension(), index.size()));
            }
            let _ = fs::remove_file(&path);
        }
    }
    let _ = fs::remove_file(&good);

    assert!(
        accepted.is_empty(),
        "{} of 128 geometry-field bit flips were accepted \
         (byte, bit, dimension, size): {:?}",
        accepted.len(),
        accepted
    );
}

#[cfg(feature = "disk")]
#[test]
fn mmap_load_rejects_size_overflow() {
    // num_vectors * dim that overflows usize when multiplied by sizeof(f32).
    let path = std::env::temp_dir().join("vanedb_mmap_overflow.bin");
    let mut data = Vec::new();
    data.extend_from_slice(&DISK_MAGIC.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&u64::MAX.to_le_bytes()); // dim huge
    data.extend_from_slice(&u64::MAX.to_le_bytes()); // num_vectors huge
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    fs::write(&path, &data).unwrap();
    assert!(matches!(
        // SAFETY: this test does not modify the file while it is mapped.
        unsafe { DiskIndex::open(&path) },
        Err(VaneError::Corrupt { .. })
    ));
    let _ = fs::remove_file(&path);
}

#[cfg(feature = "disk")]
#[test]
fn mmap_load_rejects_invalid_metric() {
    let path = std::env::temp_dir().join("vanedb_mmap_bad_metric.bin");
    // Valid header, dim=3, num=0, metric=99 (out of range)
    let mut data = Vec::new();
    data.extend_from_slice(&DISK_MAGIC.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&3u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&99u32.to_le_bytes()); // bogus metric
    data.extend_from_slice(&0u32.to_le_bytes());
    fs::write(&path, &data).unwrap();
    assert!(matches!(
        // SAFETY: this test does not modify the file while it is mapped.
        unsafe { DiskIndex::open(&path) },
        Err(VaneError::Corrupt { .. })
    ));
    let _ = fs::remove_file(&path);
}

#[cfg(feature = "disk")]
#[test]
fn mmap_search_rejects_zero_k() {
    let path = std::env::temp_dir().join("vanedb_mmap_zero_k.bin");
    let mut b = DiskIndexBuilder::new(3, Metric::L2).unwrap();
    b.add(1, &[1.0, 2.0, 3.0]).unwrap();
    b.save(&path).unwrap();
    // SAFETY: this test does not modify the file while it is mapped.
    let store = unsafe { DiskIndex::open(&path) }.unwrap();
    assert!(matches!(
        store.search(&[1.0, 2.0, 3.0], 0),
        Err(VaneError::InvalidK)
    ));
    let _ = fs::remove_file(&path);
}

/// A v2 payload whose arrays hold `count` live slots but which declares an
/// enormous `max_elements`. `load` re-expands the arrays to `max_elements`,
/// so an unbounded declaration is an allocation request from a tiny file.
fn v2_huge_max_elements(max_elements: usize) -> HnswDataMirror {
    HnswDataMirror {
        dim: 2,
        metric: 0,
        max_elements,
        m: 16,
        m_max: 16,
        m_max0: 32,
        ef_construction: 200,
        ef_search: 50,
        mult: 1.0 / (16f64).ln(),
        seed: 42,
        count: 1,
        entry_point: Some(0),
        max_level: 0,
        vectors: vec![1.0, 2.0],
        ext_ids: vec![10],
        levels: vec![0],
        neighbors: vec![vec![vec![]]],
        id_map: std::collections::HashMap::from([(10, 0)]),
    }
}

#[test]
fn hnsw_load_rejects_an_unallocatable_max_elements() {
    // ~2^40 slots: the file is a few hundred bytes, the declared expansion is
    // terabytes. Must be an error, not an abort.
    let bytes = hnsw_file_bytes(2, &v2_huge_max_elements(1 << 40));
    let p = write_tmp("huge_max_elements", &bytes);
    let err = ApproxIndex::load(&p).expect_err("must reject, not allocate");
    let msg = err.to_string();
    assert!(
        msg.contains("max_elements") || msg.contains("too large"),
        "unhelpful message: {msg}"
    );
}

#[test]
fn hnsw_load_rejects_invalid_graph_parameters() {
    // m < 2 makes mult and the level distribution meaningless; C++ validates
    // it on load and Rust did not.
    let mut data = v2_huge_max_elements(4);
    // Keep m_max/m_max0 consistent with m, or the `m_max0 != m * 2` check
    // rejects the file first and this case never reaches the `m < 2` guard.
    data.m = 1;
    data.m_max = 1;
    data.m_max0 = 2;
    let bytes = hnsw_file_bytes(2, &data);
    let p = write_tmp("bad_m", &bytes);
    assert!(
        matches!(ApproxIndex::load(&p), Err(VaneError::Corrupt { .. })),
        "m = 1 must be rejected on load"
    );

    let mut data = v2_huge_max_elements(4);
    data.ef_construction = 0;
    let bytes = hnsw_file_bytes(2, &data);
    let p = write_tmp("bad_efc", &bytes);
    assert!(
        matches!(ApproxIndex::load(&p), Err(VaneError::Corrupt { .. })),
        "ef_construction = 0 must be rejected on load"
    );
}

#[test]
fn hnsw_load_rejects_a_count_times_dim_overflow() {
    // `count * dim` sizes the vector array. v2 deliberately does not bound
    // count by max_elements (growth past capacity is supported), and dim is
    // bounded only by `dim * 4` not overflowing — so the product is an
    // unchecked multiply on two attacker-controlled values.
    //
    // 8 * 2^61 is exactly 2^64, which wraps to 0, so an EMPTY vector array
    // satisfies the length check and a few hundred bytes describe an index
    // claiming 2^61 dimensions.
    let count = 8usize;
    let dim = 1usize << (usize::BITS - 3);
    assert_eq!(
        count.wrapping_mul(dim),
        0,
        "the wrap is what this test is about"
    );

    let mut data = v2_huge_max_elements(1);
    data.dim = dim;
    data.count = count;
    data.vectors = vec![];
    data.ext_ids = (0..count as u64).collect();
    data.levels = vec![0; count];
    data.neighbors = (0..count).map(|_| vec![vec![]]).collect();
    data.id_map = (0..count as u64).map(|i| (i, i as usize)).collect();

    let bytes = hnsw_file_bytes(2, &data);
    let p = write_tmp("count_dim_overflow", &bytes);
    assert!(
        matches!(ApproxIndex::load(&p), Err(VaneError::Corrupt { .. })),
        "count * dim overflow must be rejected, not wrapped"
    );
    let _ = fs::remove_file(&p);
}

#[test]
fn hnsw_load_rejects_an_m_doubling_overflow() {
    // `m_max0 != m * 2` is itself an unchecked multiply. At m = 2^63 the
    // product wraps to 0, so a file declaring m_max0 = 0 passes the very
    // check that exists to keep the graph parameters consistent.
    let m = 1usize << (usize::BITS - 1);
    assert_eq!(m.wrapping_mul(2), 0, "the wrap is what this test is about");

    let mut data = v2_huge_max_elements(1);
    data.m = m;
    data.m_max = m;
    data.m_max0 = 0;

    let bytes = hnsw_file_bytes(2, &data);
    let p = write_tmp("m_doubling_overflow", &bytes);
    assert!(
        matches!(ApproxIndex::load(&p), Err(VaneError::Corrupt { .. })),
        "m * 2 overflow must be rejected, not wrapped"
    );
    let _ = fs::remove_file(&p);
}

#[test]
fn hnsw_load_rejects_inconsistent_graph_structure() {
    for case in [
        "negative_level",
        "missing_layer",
        "extra_layer",
        "max_level",
        "self_link",
        "duplicate_link",
        "degree",
        "neighbor_layer",
    ] {
        let mut data = v1_full_capacity_payload();
        match case {
            "negative_level" => data.levels[0] = -1,
            "missing_layer" => data.neighbors[0].clear(),
            "extra_layer" => data.neighbors[0].push(vec![]),
            "max_level" => data.max_level = 1,
            "self_link" => data.neighbors[0][0].push(0),
            "duplicate_link" => data.neighbors[0][0].push(1),
            "degree" => data.neighbors[0][0] = vec![1; 5],
            "neighbor_layer" => {
                data.levels[0] = 1;
                data.max_level = 1;
                data.neighbors[0].push(vec![1]);
            }
            _ => unreachable!(),
        }
        let p = write_tmp(case, &hnsw_file_bytes(1, &data));
        assert!(
            matches!(ApproxIndex::load(&p), Err(VaneError::Corrupt { .. })),
            "accepted {case}"
        );
        fs::remove_file(p).unwrap();
    }
}

/// A complete, self-consistent `VNDB` file. Every derived length agrees, so
/// only the deliberately-planted defect can make a loader reject it — a
/// short file is caught by the truncation check first, which is how a header
/// guard can look tested when it never runs.
#[cfg(feature = "disk")]
fn disk_file_bytes(magic: u32, version: u32, dim: usize, ids: &[u64], vectors: &[f32]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&magic.to_le_bytes());
    data.extend_from_slice(&version.to_le_bytes());
    data.extend_from_slice(&(dim as u64).to_le_bytes());
    data.extend_from_slice(&(ids.len() as u64).to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // metric: L2
    data.extend_from_slice(&0u32.to_le_bytes()); // reserved
    for id in ids {
        data.extend_from_slice(&id.to_le_bytes());
    }
    for v in vectors {
        data.extend_from_slice(&v.to_le_bytes());
    }
    data
}

#[cfg(feature = "disk")]
#[test]
fn mmap_load_rejects_duplicate_ids() {
    // Both builders reject duplicates on write, so no vanedb writer produces
    // this — but VNDB is the shared cross-engine format, so the loader is
    // what has to distrust it. Last-write-wins silently made `size()`
    // overcount, `get` return the wrong row, and `search` emit one id twice.
    let bytes = disk_file_bytes(
        DISK_MAGIC,
        1,
        2,
        &[7, 7, 9],
        &[1.0, 0.0, 0.0, 1.0, 5.0, 4.0],
    );
    let p = write_tmp("mmap_duplicate_ids", &bytes);
    assert!(
        // SAFETY: this test does not modify the file while it is mapped.
        matches!(
            unsafe { DiskIndex::open(&p) },
            Err(VaneError::Corrupt { .. })
        ),
        "a VNDB file with duplicate ids must be rejected"
    );
    let _ = fs::remove_file(&p);
}

#[cfg(feature = "disk")]
#[test]
fn mmap_load_rejects_invalid_magic() {
    // The only previous "bad file" test wrote 7 bytes, which the length floor
    // rejects before the magic is ever compared. Disabling the magic check
    // left the whole suite green.
    let bytes = disk_file_bytes(
        u32::from_le_bytes(*b"BDNV"),
        1,
        2,
        &[1, 2],
        &[1.0, 0.0, 0.0, 1.0],
    );
    let p = write_tmp("mmap_bad_magic", &bytes);
    // SAFETY: this test does not modify the file while it is mapped.
    let err = match unsafe { DiskIndex::open(&p) } {
        Ok(_) => panic!("a file with the wrong magic must not open"),
        Err(e) => e,
    };
    assert!(format!("{err}").contains("magic"), "got: {err}");
    let _ = fs::remove_file(&p);
}

#[test]
fn hnsw_load_accepts_the_checked_in_v1_fixture() {
    // Loads bytes committed to the repository, not bytes this test just
    // encoded. The mirror-based test above re-serializes with today's bincode,
    // so mirror and encoder change together and it stays green even if a real
    // file from an earlier release has stopped loading. This one cannot: the
    // bytes are frozen.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hnsw_v1.bin");
    let idx = ApproxIndex::load(&path).expect("the committed v1 fixture must load");
    assert_eq!(idx.size(), 2);
    assert_eq!(idx.capacity(), 4);
    assert_eq!(idx.get_vector(10).unwrap(), vec![1.0, 0.0]);
    assert_eq!(idx.get_vector(20).unwrap(), vec![0.0, 1.0]);
    let hits = idx.search(&[1.0, 0.1], 1).unwrap();
    assert_eq!(hits[0].id, 10);
}
