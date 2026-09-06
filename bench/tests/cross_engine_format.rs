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
