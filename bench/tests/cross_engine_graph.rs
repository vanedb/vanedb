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

/// The test above deliberately builds a complete graph (`M = N + 4`) so every
/// live ID can be compared exhaustively. That makes its adjacency trivial:
/// every node links to every other, the layer-0 cap of `2M` and the upper-layer
/// cap of `M` can never bind, and `1/ln(68)` puts an expected 0.94 nodes above
/// layer 0 — so the multi-level structure is effectively absent.
///
/// Byte-identical round-trip across engines therefore proved that ids, vectors
/// and tombstones survive, but said almost nothing about neighbour lists. This
/// builds the graph the other test cannot: realistic `M`, degree caps that
/// bind, reverse-link truncation exercised, and enough nodes that upper layers
/// are populated. Cross-loading it and re-saving byte-for-byte is a real
/// topology check.
#[test]
fn a_sparse_multi_level_graph_survives_cross_loading_byte_for_byte() {
    const SPARSE_N: usize = 256;
    const M: usize = 16;

    for metric in 0..3 {
        for source_rust in [true, false] {
            let dir = std::env::temp_dir().join(format!(
                "vanedb-cross-sparse-{}-{metric}-{source_rust}",
                std::process::id()
            ));
            fs::create_dir_all(&dir).unwrap();
            let original = dir.join("original.vndb");
            let roundtrip = dir.join("roundtrip.vndb");

            let rs = Rust(unsafe { rust::vanedb_rs_index_new(DIM, metric, SPARSE_N, M, 200, 42) });
            let cpp = Cpp(unsafe { ffi::vanedb_cpp_index_new(DIM, metric, SPARSE_N, M, 200, 42) });
            assert!(!rs.0.is_null() && !cpp.0.is_null());

            for i in 0..SPARSE_N {
                let v = vector(i % 64);
                let rc = unsafe {
                    if source_rust {
                        rust::vanedb_rs_index_add(rs.0, i as u64, v.as_ptr())
                    } else {
                        ffi::vanedb_cpp_index_add(cpp.0, i as u64, v.as_ptr())
                    }
                };
                assert_eq!(rc, 0, "add {i} must succeed");
            }

            let rc = unsafe {
                if source_rust {
                    rust::vanedb_rs_index_save(rs.0, path(&original).as_ptr())
                } else {
                    ffi::vanedb_cpp_index_save(cpp.0, path(&original).as_ptr())
                }
            };
            assert_eq!(rc, 0, "save must succeed");
            let bytes = fs::read(&original).unwrap();

            // A complete graph on 256 nodes at 4 bytes per neighbour slot would
            // dwarf this. Staying well under proves the degree caps bound the
            // adjacency, which is the property the dense test cannot show.
            let complete_lower_bound = SPARSE_N * (SPARSE_N - 1) * 8;
            assert!(
                bytes.len() < complete_lower_bound / 4,
                "graph looks complete ({} bytes); M is not binding and this test \
                 would prove nothing about adjacency",
                bytes.len()
            );

            // Each engine must read the other's sparse graph and reproduce it
            // exactly. A reader that mis-ordered or truncated a neighbour list
            // fails here; the dense case cannot see either error.
            let rs_loaded = Rust(unsafe { rust::vanedb_rs_index_load(path(&original).as_ptr()) });
            let cpp_loaded = Cpp(unsafe { ffi::vanedb_cpp_index_load(path(&original).as_ptr()) });
            assert!(
                !rs_loaded.0.is_null() && !cpp_loaded.0.is_null(),
                "both engines must load a sparse graph written by {}",
                if source_rust { "Rust" } else { "C++" }
            );

            assert_eq!(
                unsafe { rust::vanedb_rs_index_save(rs_loaded.0, path(&roundtrip).as_ptr()) },
                0
            );
            assert_eq!(
                fs::read(&roundtrip).unwrap(),
                bytes,
                "Rust must reproduce a sparse graph byte for byte"
            );
            assert_eq!(
                unsafe { ffi::vanedb_cpp_index_save(cpp_loaded.0, path(&roundtrip).as_ptr()) },
                0
            );
            assert_eq!(
                fs::read(&roundtrip).unwrap(),
                bytes,
                "C++ must reproduce a sparse graph byte for byte"
            );

            let _ = fs::remove_dir_all(&dir);
        }
    }
}
