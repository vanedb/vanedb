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

    // Header bits, then a few payload bits, then the two length edges.
    let mut cases: Vec<(String, Vec<u8>)> = Vec::new();
    for byte in (0..32usize).chain([32, 40, 80, 100]) {
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
    let payload_accepted = [32usize, 40, 80, 100]
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
