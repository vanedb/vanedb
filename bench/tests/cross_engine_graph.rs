//! Preserve actual engine-written graphs across languages, including tombstones.
use std::ffi::CString;
use std::fs;
use std::path::Path;
use vanedb_bench::ffi;
use vanedb_capi as rust;

const DIM: usize = 8;
const N: usize = 64;

struct Cpp(*mut std::ffi::c_void);
impl Drop for Cpp {
    fn drop(&mut self) {
        unsafe { ffi::vanedb_cpp_index_free(self.0) }
    }
}
struct Rust(*mut rust::vanedb_rs_index);
impl Drop for Rust {
    fn drop(&mut self) {
        unsafe { rust::vanedb_rs_index_free(self.0) }
    }
}
fn path(path: &Path) -> CString {
    CString::new(path.to_str().unwrap()).unwrap()
}
fn vector(i: usize) -> Vec<f32> {
    (0..DIM)
        .map(|d| (((i + 1) * (d + 5) * 13) % 101) as f32 / 101.0)
        .collect()
}
fn agree(rs: &Rust, cpp: &Cpp, count: usize) {
    for query in [vector(0), vector(17), vector(60)] {
        let (mut rs_ids, mut rs_d) = (vec![0; N + 4], vec![0.; N + 4]);
        let (mut cpp_ids, mut cpp_d) = (vec![0; N + 4], vec![0.; N + 4]);
        let (nr, nc) = unsafe {
            (
                rust::vanedb_rs_index_search(
                    rs.0,
                    query.as_ptr(),
                    N + 4,
                    N + 4,
                    rs_ids.as_mut_ptr(),
                    rs_d.as_mut_ptr(),
                ),
                ffi::vanedb_cpp_index_search(
                    cpp.0,
                    query.as_ptr(),
                    N + 4,
                    N + 4,
                    cpp_ids.as_mut_ptr(),
                    cpp_d.as_mut_ptr(),
                ),
            )
        };
        assert_eq!(nr, count);
        assert_eq!(nr, nc);
        // SIMD accumulation can permute nearly tied ranks. Require the same
        // IDs, each ID's distance, and the distance at every rank to agree.
        for (a, b) in rs_d[..nr].iter().zip(&cpp_d[..nc]) {
            assert!((a - b).abs() < 1e-5);
        }
        let mut rs_hits: Vec<_> = rs_ids[..nr].iter().zip(&rs_d[..nr]).collect();
        let mut cpp_hits: Vec<_> = cpp_ids[..nc].iter().zip(&cpp_d[..nc]).collect();
        rs_hits.sort_unstable_by_key(|hit| hit.0);
        cpp_hits.sort_unstable_by_key(|hit| hit.0);
        for ((rs_id, rs_distance), (cpp_id, cpp_distance)) in rs_hits.iter().zip(&cpp_hits) {
            assert_eq!(rs_id, cpp_id);
            assert!((*rs_distance - *cpp_distance).abs() < 1e-5);
        }
    }
}

#[test]
fn engine_written_graphs_cross_load_preserve_topology_and_remain_mutable() {
    for metric in 0..3 {
        for source_rust in [true, false] {
            let dir = std::env::temp_dir().join(format!(
                "vanedb-cross-graph-{}-{metric}-{source_rust}",
                std::process::id()
            ));
            fs::create_dir_all(&dir).unwrap();
            let original = dir.join("original.vndb");
            let roundtrip = dir.join("roundtrip.vndb");
            // Keep this small graph fully connected so every live ID can be
            // compared. Recall of sparse graphs has separate workload tests.
            let rs = Rust(unsafe {
                rust::vanedb_rs_index_new(DIM, metric, N + 4, N + 4, N + 4, u64::MAX)
            });
            let cpp =
                Cpp(unsafe { ffi::vanedb_cpp_index_new(DIM, metric, N + 4, N + 4, N + 4, 42) });
            assert!(!rs.0.is_null() && !cpp.0.is_null());
            for i in 0..N {
                // Include IDs beyond f64's exact range and the u64 endpoint.
                let id = u64::MAX - i as u64;
                let v = vector(i);
                assert_eq!(
                    unsafe {
                        if source_rust {
                            rust::vanedb_rs_index_add(rs.0, id, v.as_ptr())
                        } else {
                            ffi::vanedb_cpp_index_add(cpp.0, id, v.as_ptr())
                        }
                    },
                    0
                );
            }
            let count = if source_rust {
                // Remove the entry's possible ID and reuse an existing identity.
                assert_eq!(unsafe { rust::vanedb_rs_index_remove(rs.0, u64::MAX) }, 0);
                assert_eq!(
                    unsafe { rust::vanedb_rs_index_remove(rs.0, u64::MAX - 1) },
                    0
                );
                assert_eq!(
                    unsafe { rust::vanedb_rs_index_add(rs.0, u64::MAX - 1, vector(N).as_ptr()) },
                    0
                );
                N - 1
            } else {
                N
            };
            assert_eq!(
                unsafe {
                    if source_rust {
                        rust::vanedb_rs_index_save(rs.0, path(&original).as_ptr())
                    } else {
                        ffi::vanedb_cpp_index_save(cpp.0, path(&original).as_ptr())
                    }
                },
                0
            );
            let bytes = fs::read(&original).unwrap();
            let rs_loaded = Rust(unsafe { rust::vanedb_rs_index_load(path(&original).as_ptr()) });
            let cpp_loaded = Cpp(unsafe { ffi::vanedb_cpp_index_load(path(&original).as_ptr()) });
            assert!(!rs_loaded.0.is_null() && !cpp_loaded.0.is_null());
            // Search changes ef, so prove exact preservation before running it.
            assert_eq!(
                unsafe { rust::vanedb_rs_index_save(rs_loaded.0, path(&roundtrip).as_ptr()) },
                0
            );
            assert_eq!(fs::read(&roundtrip).unwrap(), bytes);
            assert_eq!(
                unsafe { ffi::vanedb_cpp_index_save(cpp_loaded.0, path(&roundtrip).as_ptr()) },
                0
            );
            assert_eq!(fs::read(&roundtrip).unwrap(), bytes);
            agree(&rs_loaded, &cpp_loaded, count);
            // Each imported graph must still accept an insertion. Save it and
            // have the other engine check that exact graph again.
            for mutate_rust in [true, false] {
                assert_eq!(
                    unsafe {
                        if mutate_rust {
                            rust::vanedb_rs_index_add(rs_loaded.0, 7, vector(N + 1).as_ptr())
                        } else {
                            ffi::vanedb_cpp_index_add(cpp_loaded.0, 7, vector(N + 1).as_ptr())
                        }
                    },
                    0
                );
                assert_eq!(
                    unsafe {
                        if mutate_rust {
                            rust::vanedb_rs_index_save(rs_loaded.0, path(&roundtrip).as_ptr())
                        } else {
                            ffi::vanedb_cpp_index_save(cpp_loaded.0, path(&roundtrip).as_ptr())
                        }
                    },
                    0
                );
                let other_rs =
                    Rust(unsafe { rust::vanedb_rs_index_load(path(&roundtrip).as_ptr()) });
                let other_cpp =
                    Cpp(unsafe { ffi::vanedb_cpp_index_load(path(&roundtrip).as_ptr()) });
                assert!(!other_rs.0.is_null() && !other_cpp.0.is_null());
                agree(&other_rs, &other_cpp, count + 1);
            }
            fs::remove_dir_all(dir).unwrap();
        }
    }
}
