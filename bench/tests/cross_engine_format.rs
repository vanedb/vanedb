//! Each engine must read the other's `DiskIndex` file.
//!
//! The engines write byte-identical headers and payloads, and this is what
//! keeps that true. Every other conformance fixture has each engine check
//! itself against shared expectations; this is the only test where one engine
//! consumes the other's output.

use std::ffi::CString;
use vanedb_bench::ffi;

const DIM: usize = 8;
const N: usize = 64;
const K: usize = 5;

fn workload() -> (Vec<u64>, Vec<f32>) {
    let ids: Vec<u64> = (0..N as u64).collect();
    let vectors: Vec<f32> = (0..N)
        .flat_map(|i| (0..DIM).map(move |d| ((i * 7 + d * 3) % 23) as f32))
        .collect();
    (ids, vectors)
}

/// Every metric the header can encode. Only L2 was ever cross-loaded, so a
/// divergence in the cosine or dot encoding would not have been noticed.
const METRICS: [(u32, &str); 3] = [(0, "l2"), (1, "cosine"), (2, "dot")];

fn query() -> Vec<f32> {
    (0..DIM).map(|d| (d * 3 % 23) as f32).collect()
}

fn agree(a_ids: &[u64], a_d: &[f32], b_ids: &[u64], b_d: &[f32], what: &str) {
    assert_eq!(a_ids, b_ids, "{what}: neighbour ids differ across engines");
    for (x, y) in a_d.iter().zip(b_d.iter()) {
        assert!((x - y).abs() < 1e-5, "{what}: distances differ: {x} vs {y}");
    }
}

#[test]
fn cpp_reads_a_file_written_by_rust() {
    for (metric, name) in METRICS {
        cpp_reads_rust_with_metric(metric, name);
    }
}

fn cpp_reads_rust_with_metric(metric: u32, name: &str) {
    let (ids, vectors) = workload();
    let q = query();
    let file = format!("cross_rs_to_cpp_{name}.vndb");
    let path = CString::new(file.clone()).unwrap();
    let (mut rs_ids, mut rs_d) = ([0u64; K], [0f32; K]);
    let (mut cpp_ids, mut cpp_d) = ([0u64; K], [0f32; K]);

    unsafe {
        assert_eq!(
            ffi::vanedb_rs_disk_build(
                path.as_ptr(),
                DIM,
                metric,
                ids.as_ptr(),
                vectors.as_ptr(),
                N
            ),
            0,
            "rust build failed"
        );
        let rs = ffi::vanedb_rs_disk_open(path.as_ptr());
        assert!(!rs.is_null(), "rust cannot open its own file");
        let n_rs =
            ffi::vanedb_rs_disk_search(rs, q.as_ptr(), K, rs_ids.as_mut_ptr(), rs_d.as_mut_ptr());
        ffi::vanedb_rs_disk_free(rs);

        let cpp = ffi::vanedb_cpp_disk_open(path.as_ptr());
        assert!(
            !cpp.is_null(),
            "C++ rejected a {name} file written by Rust — the formats have diverged"
        );
        let n_cpp = ffi::vanedb_cpp_disk_search(
            cpp,
            q.as_ptr(),
            K,
            cpp_ids.as_mut_ptr(),
            cpp_d.as_mut_ptr(),
        );
        ffi::vanedb_cpp_disk_free(cpp);

        assert_eq!(n_rs, n_cpp, "result counts differ");
        agree(
            &rs_ids,
            &rs_d,
            &cpp_ids,
            &cpp_d,
            &format!("rust -> cpp ({name})"),
        );
    }
    let _ = std::fs::remove_file(&file);
}

#[test]
fn rust_reads_a_file_written_by_cpp() {
    for (metric, name) in METRICS {
        rust_reads_cpp_with_metric(metric, name);
    }
}

fn rust_reads_cpp_with_metric(metric: u32, name: &str) {
    let (ids, vectors) = workload();
    let q = query();
    let file = format!("cross_cpp_to_rs_{name}.vndb");
    let path = CString::new(file.clone()).unwrap();
    let (mut rs_ids, mut rs_d) = ([0u64; K], [0f32; K]);
    let (mut cpp_ids, mut cpp_d) = ([0u64; K], [0f32; K]);

    unsafe {
        assert_eq!(
            ffi::vanedb_cpp_disk_build(
                path.as_ptr(),
                DIM,
                metric,
                ids.as_ptr(),
                vectors.as_ptr(),
                N
            ),
            0,
            "cpp build failed"
        );
        let cpp = ffi::vanedb_cpp_disk_open(path.as_ptr());
        assert!(!cpp.is_null(), "C++ cannot open its own file");
        let n_cpp = ffi::vanedb_cpp_disk_search(
            cpp,
            q.as_ptr(),
            K,
            cpp_ids.as_mut_ptr(),
            cpp_d.as_mut_ptr(),
        );
        ffi::vanedb_cpp_disk_free(cpp);

        let rs = ffi::vanedb_rs_disk_open(path.as_ptr());
        assert!(
            !rs.is_null(),
            "Rust rejected a file written by C++ — the formats have diverged"
        );
        let n_rs =
            ffi::vanedb_rs_disk_search(rs, q.as_ptr(), K, rs_ids.as_mut_ptr(), rs_d.as_mut_ptr());
        ffi::vanedb_rs_disk_free(rs);

        assert_eq!(n_cpp, n_rs, "result counts differ");
        agree(&cpp_ids, &cpp_d, &rs_ids, &rs_d, "cpp -> rust");
    }
    let _ = std::fs::remove_file(&file);
}

/// Both engines must reject exactly the same corrupt files.
///
/// A shared format is only shared if its readers agree on what is valid.
/// Cross-loading proves they agree on *good* files; nothing proved they agree
/// on bad ones, and they did not: both carried a one-sided `file_size <
/// expected` check that accepted a header understating the geometry, and
/// fixing one engine without the other would have been a silent divergence —
/// a file one engine reads and the other refuses.
///
/// The sweep covers every single-bit flip in the 32-byte header plus a sample
/// of payload bits, so it also pins what the format deliberately does *not*
/// catch: with no checksum, a flipped payload bit is valid to both engines,
/// and a metric flip that lands on another defined metric is a valid file.
/// `DiskIndex::open`'s Integrity section says exactly this; the counts here
/// are what make that statement checkable.
#[test]
fn both_engines_accept_and_reject_exactly_the_same_files() {
    let (ids, vectors) = workload();
    let dir = std::env::temp_dir().join(format!("vanedb-agree-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let good = dir.join("good.vndb");
    let c_good = CString::new(good.to_str().unwrap()).unwrap();
    // SAFETY: buffers outlive the call and the lengths match `ids`/`vectors`.
    assert_eq!(
        unsafe {
            ffi::vanedb_rs_disk_build(c_good.as_ptr(), DIM, 0, ids.as_ptr(), vectors.as_ptr(), N)
        },
        0
    );
    let original = std::fs::read(&good).unwrap();

    // Header bits, then payload bits from BOTH payload regions, then the two
    // length edges. The ids occupy [32, 32 + N*8) and the vectors follow, so
    // sampling only low offsets tests ids and never a vector component.
    let ids_end = 32 + N * 8;
    let sampled: Vec<usize> = vec![32, 40, 80, 100, ids_end, ids_end + 4, ids_end + 64];
    let mut cases: Vec<(String, Vec<u8>)> = Vec::new();
    for byte in (0..32usize).chain(sampled.iter().copied()) {
        for bit in 0..8u32 {
            let mut bytes = original.clone();
            bytes[byte] ^= 1 << bit;
            cases.push((format!("flip_{byte}_{bit}"), bytes));
        }
    }
    let mut longer = original.clone();
    longer.push(0);
    cases.push(("trailing_byte".into(), longer));
    cases.push((
        "truncated_byte".into(),
        original[..original.len() - 1].to_vec(),
    ));
    cases.push(("pristine".into(), original.clone()));

    let mut disagreements = Vec::new();
    let mut verdict = std::collections::HashMap::new();
    for (name, bytes) in &cases {
        let path = dir.join(format!("{name}.vndb"));
        std::fs::write(&path, bytes).unwrap();
        let c_path = CString::new(path.to_str().unwrap()).unwrap();

        // SAFETY: this test owns the file and never mutates it while mapped.
        let rust_ok = unsafe { vanedb::DiskIndex::open(&path) }.is_ok();
        // SAFETY: same file; the handle is freed before the next iteration.
        let cpp_handle = unsafe { ffi::vanedb_cpp_disk_open(c_path.as_ptr()) };
        let cpp_ok = !cpp_handle.is_null();
        if cpp_ok {
            // SAFETY: non-null handle from the matching constructor.
            unsafe { ffi::vanedb_cpp_disk_free(cpp_handle) };
        }

        if rust_ok != cpp_ok {
            disagreements.push(format!("{name}: rust={rust_ok} cpp={cpp_ok}"));
        }
        verdict.insert(name.clone(), rust_ok);
        let _ = std::fs::remove_file(&path);
    }
    // Cleanup happens before the assertions on purpose: a failing run should
    // leave nothing behind, and the failure message carries everything needed.
    let _ = std::fs::remove_dir_all(&dir);

    // The claim this test exists for.
    assert!(
        disagreements.is_empty(),
        "the engines disagree on {} of {} files: {disagreements:?}",
        disagreements.len(),
        cases.len()
    );

    // What the readers must reject: every bit of magic, version, dim and
    // num_vectors, and the reserved word. `dim` and `num_vectors` are the
    // fields the payload length is derived from, so a flip there makes the
    // header disagree with the file in one direction or the other -- which is
    // exactly what a one-sided length check missed.
    let structural = (0..8usize).chain(8..24).chain(28..32);
    for byte in structural {
        for bit in 0..8u32 {
            let name = format!("flip_{byte}_{bit}");
            assert!(
                !verdict[&name],
                "{name} changes the file's shape and must be rejected by both engines"
            );
        }
    }

    // The metric field is the one header byte where a flip can produce a
    // structurally valid file: 0 -> 1 and 0 -> 2 are other defined metrics.
    // Nothing in the format records which one was intended, so both engines
    // accept them and answer with a different distance function.
    assert!(verdict["flip_24_0"], "L2 -> Cosine is a valid file");
    assert!(verdict["flip_24_1"], "L2 -> Dot is a valid file");
    for bit in 2..8u32 {
        assert!(
            !verdict[&format!("flip_24_{bit}")],
            "an undefined metric value must be rejected"
        );
    }

    // Both length edges, which is the equality check itself.
    assert!(verdict["pristine"]);
    assert!(
        !verdict["truncated_byte"],
        "a byte short of its declared geometry"
    );
    assert!(
        !verdict["trailing_byte"],
        "a byte past its declared geometry -- the direction `<` missed"
    );

    // And what the format deliberately does not protect: it carries no
    // checksum, so at least one payload flip is a valid file to both engines.
    // `DiskIndex::open`'s Integrity section says so; this keeps that honest
    // rather than letting the doc drift into promising more than it delivers.
    let payload_accepted = sampled
        .iter()
        .flat_map(|byte| (0..8u32).map(move |bit| format!("flip_{byte}_{bit}")))
        .filter(|name| verdict[name])
        .count();
    assert!(
        payload_accepted > 0,
        "no payload flip was accepted -- if the format gained a checksum, \
         the Integrity section in disk.rs must be updated to say so"
    );
}

/// Not every component flip survives, which is why the Integrity section says a
/// component flip is "usually" returned rather than "returned".
///
/// Constructed rather than sampled: whether a flip produces an infinity depends
/// on the component's exponent bits, so a sweep over an arbitrary workload may
/// or may not contain one. `1.0f32` is `0x3F800000`; setting bit 30 fills the
/// exponent and gives `+inf`, which both readers reject.
#[test]
fn a_component_flip_that_produces_an_infinity_is_rejected_by_both() {
    let dir = std::env::temp_dir().join(format!("vanedb-inf-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let good = dir.join("good.vndb");
    let c_good = CString::new(good.to_str().unwrap()).unwrap();
    let ids = [7u64];
    let vectors = [1.0f32; DIM];
    // SAFETY: both buffers hold exactly the lengths declared.
    assert_eq!(
        unsafe {
            ffi::vanedb_rs_disk_build(c_good.as_ptr(), DIM, 0, ids.as_ptr(), vectors.as_ptr(), 1)
        },
        0
    );

    let mut bytes = std::fs::read(&good).unwrap();
    let first_component = 32 + 8; // one id, then the vectors
    assert_eq!(
        f32::from_le_bytes(
            bytes[first_component..first_component + 4]
                .try_into()
                .unwrap()
        ),
        1.0
    );
    bytes[first_component + 3] ^= 0x40; // bit 30: 0x3F800000 -> 0x7F800000
    assert!(f32::from_le_bytes(
        bytes[first_component..first_component + 4]
            .try_into()
            .unwrap()
    )
    .is_infinite());

    let bad = dir.join("inf.vndb");
    std::fs::write(&bad, &bytes).unwrap();
    let c_bad = CString::new(bad.to_str().unwrap()).unwrap();

    // SAFETY: this test owns the file and never mutates it while mapped.
    assert!(
        unsafe { vanedb::DiskIndex::open(&bad) }.is_err(),
        "an infinite component must be rejected"
    );
    // SAFETY: same file.
    let handle = unsafe { ffi::vanedb_cpp_disk_open(c_bad.as_ptr()) };
    if !handle.is_null() {
        // SAFETY: non-null handle from the matching constructor.
        unsafe { ffi::vanedb_cpp_disk_free(handle) };
        panic!("the C++ reader accepted an infinite component");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// The same agreement, on an empty store.
///
/// Separate because an empty store is 32 bytes whatever its `dim`, so
/// `expected` is 32 for every dimension and the length check cannot see the
/// field at all. That is the shape the non-empty sweep structurally cannot
/// reach, and it is where the two readers last disagreed: C++ bounds `dim`
/// unconditionally, while Rust derived its only bound from a product that is
/// zero when `num_vectors` is zero, so `dim = 2^62` was a file one engine read
/// and the other refused.
#[test]
fn both_engines_agree_on_an_empty_store_too() {
    let dir = std::env::temp_dir().join(format!("vanedb-agree-empty-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let good = dir.join("empty.vndb");
    let c_good = CString::new(good.to_str().unwrap()).unwrap();
    // SAFETY: n is 0, so the null id/vector pointers are never read.
    assert_eq!(
        unsafe {
            ffi::vanedb_rs_disk_build(c_good.as_ptr(), 4, 0, std::ptr::null(), std::ptr::null(), 0)
        },
        0
    );
    let original = std::fs::read(&good).unwrap();
    assert_eq!(original.len(), 32, "an empty store is header-only");

    let mut disagreements = Vec::new();
    for byte in 8..24usize {
        for bit in 0..8u32 {
            let mut bytes = original.clone();
            bytes[byte] ^= 1 << bit;
            let path = dir.join(format!("e_{byte}_{bit}.vndb"));
            std::fs::write(&path, &bytes).unwrap();
            let c_path = CString::new(path.to_str().unwrap()).unwrap();

            // SAFETY: this test owns the file and never mutates it while mapped.
            let rust_ok = unsafe { vanedb::DiskIndex::open(&path) }.is_ok();
            // SAFETY: same file; the handle is freed before the next iteration.
            let handle = unsafe { ffi::vanedb_cpp_disk_open(c_path.as_ptr()) };
            let cpp_ok = !handle.is_null();
            if cpp_ok {
                // SAFETY: non-null handle from the matching constructor.
                unsafe { ffi::vanedb_cpp_disk_free(handle) };
            }
            if rust_ok != cpp_ok {
                disagreements.push(format!(
                    "byte {byte} bit {bit}: rust={rust_ok} cpp={cpp_ok}"
                ));
            }
            let _ = std::fs::remove_file(&path);
        }
    }
    assert!(
        disagreements.is_empty(),
        "the engines disagree on an empty store: {disagreements:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
