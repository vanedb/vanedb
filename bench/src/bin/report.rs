use std::hint::black_box;
use std::time::Instant;
use vanedb_bench::{ffi, ground_truth, workloads};

const DIM: usize = 128;
const N: usize = 10_000;
const K: usize = 10;

/// Median one-call latency for two engines, sampled interleaved (a, b, a, b…)
/// after a joint warmup. Interleaving makes frequency ramp, core migration,
/// and cache drift hit both engines equally — measuring one engine's block
/// first and the other's second biases the verdict toward whichever runs on
/// the warmer machine state (observed flipping store_search vs criterion).
fn median_pair_ns(mut a: impl FnMut(), mut b: impl FnMut()) -> (u128, u128) {
    const WARMUP: usize = 200;
    const SAMPLES: usize = 501;
    for _ in 0..WARMUP {
        a();
        b();
    }
    let mut ta = Vec::with_capacity(SAMPLES);
    let mut tb = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t = Instant::now();
        a();
        ta.push(t.elapsed().as_nanos());
        let t = Instant::now();
        b();
        tb.push(t.elapsed().as_nanos());
    }
    ta.sort_unstable();
    tb.sort_unstable();
    (ta[SAMPLES / 2], tb[SAMPLES / 2])
}

const N_QUERIES: usize = 100;

fn main() {
    let w = workloads::generate(99, DIM, N, N_QUERIES);
    let q = &w.queries[0..DIM];
    let mut out = String::from("# VaneDB Benchmark Results\n\n");
    out.push_str(&format!(
        "Engines: vanedb-cpp (CMake Release) and vanedb (Rust), monorepo {}.\n",
        env!("VANEDB_MONOREPO_REV"),
    ));
    out.push_str(&format!(
        "Workload: dim={DIM}, n={N}, k={K}, L2. Latencies are medians of 501 \
         interleaved paired samples (one query) after a joint warmup; recall is \
         averaged over {N_QUERIES} queries. Both engines' data stays resident \
         in one process (interleaved construction).\n\n"
    ));
    out.push_str(&format!(
        "Covers l2_sq, store_search, and hnsw_search + recall@{K} only; \
         hnsw_build and mmap_search live in the criterion suite (see README).\n\n"
    ));
    out.push_str("| Op | C++ (ns) | Rust (ns) | ratio (rs/cpp) |\n|---|---:|---:|---:|\n");

    unsafe {
        // Distance — timed in batches of 1000: a single ~10 ns call is below
        // Instant::now() resolution.
        let a = &w.vectors[0..DIM];
        let b = &w.vectors[DIM..2 * DIM];
        let (cpp, rs) = median_pair_ns(
            || {
                for _ in 0..1000 {
                    black_box(ffi::vanedb_cpp_l2_sq(a.as_ptr(), b.as_ptr(), DIM));
                }
            },
            || {
                for _ in 0..1000 {
                    black_box(ffi::vanedb_rs_l2_sq(a.as_ptr(), b.as_ptr(), DIM));
                }
            },
        );
        // Ratio from the raw batch totals — dividing to per-call ns first
        // truncates ~13.9 vs ~15.0 into 13 vs 15 and distorts the ratio.
        let ratio = rs as f64 / cpp as f64;
        let (cpp, rs) = (cpp / 1000, rs / 1000);
        out.push_str(&format!("| l2_sq | {cpp} | {rs} | {ratio:.2} |\n"));

        // Store search. Setup asserts keep a failed engine from benchmarking
        // as infinitely fast.
        let sc = ffi::vanedb_cpp_store_new(DIM, 0);
        let sr = ffi::vanedb_rs_store_new(DIM, 0);
        assert!(!sc.is_null() && !sr.is_null(), "store_new failed");
        for i in 0..N {
            assert_eq!(
                ffi::vanedb_cpp_store_add(sc, w.ids[i], w.vectors[i * DIM..].as_ptr()),
                0
            );
            assert_eq!(
                ffi::vanedb_rs_store_add(sr, w.ids[i], w.vectors[i * DIM..].as_ptr()),
                0
            );
        }
        let mut ids_c = [0u64; K];
        let mut ds_c = [0f32; K];
        let mut ids_r = [0u64; K];
        let mut ds_r = [0f32; K];
        assert_eq!(
            ffi::vanedb_cpp_store_search(sc, q.as_ptr(), K, ids_c.as_mut_ptr(), ds_c.as_mut_ptr()),
            K
        );
        assert_eq!(
            ffi::vanedb_rs_store_search(sr, q.as_ptr(), K, ids_r.as_mut_ptr(), ds_r.as_mut_ptr()),
            K
        );
        let (cpp, rs) = median_pair_ns(
            || {
                ffi::vanedb_cpp_store_search(
                    sc,
                    q.as_ptr(),
                    K,
                    ids_c.as_mut_ptr(),
                    ds_c.as_mut_ptr(),
                );
            },
            || {
                ffi::vanedb_rs_store_search(
                    sr,
                    q.as_ptr(),
                    K,
                    ids_r.as_mut_ptr(),
                    ds_r.as_mut_ptr(),
                );
            },
        );
        out.push_str(&format!(
            "| store_search | {cpp} | {rs} | {:.2} |\n",
            rs as f64 / cpp as f64
        ));
        ffi::vanedb_cpp_store_free(sc);
        ffi::vanedb_rs_store_free(sr);

        // HNSW search + recall@k vs brute-force truth, averaged over all queries
        let hc = ffi::vanedb_cpp_hnsw_new(DIM, 0, N, 16, 200, 7);
        let hr = ffi::vanedb_rs_hnsw_new(DIM, 0, N, 16, 200, 7);
        assert!(!hc.is_null() && !hr.is_null(), "hnsw_new failed");
        for i in 0..N {
            assert_eq!(
                ffi::vanedb_cpp_hnsw_add(hc, w.ids[i], w.vectors[i * DIM..].as_ptr()),
                0
            );
            assert_eq!(
                ffi::vanedb_rs_hnsw_add(hr, w.ids[i], w.vectors[i * DIM..].as_ptr()),
                0
            );
        }
        let mut ic = [0u64; K];
        let mut dc = [0f32; K];
        let mut ir = [0u64; K];
        let mut dr = [0f32; K];
        let (mut rec_c, mut rec_r) = (0.0f32, 0.0f32);
        for qi in 0..N_QUERIES {
            let query = &w.queries[qi * DIM..(qi + 1) * DIM];
            let truth = ground_truth::brute_force_topk(&w.vectors, &w.ids, DIM, query, K);
            let nc = ffi::vanedb_cpp_hnsw_search(
                hc,
                query.as_ptr(),
                K,
                50,
                ic.as_mut_ptr(),
                dc.as_mut_ptr(),
            );
            let nr = ffi::vanedb_rs_hnsw_search(
                hr,
                query.as_ptr(),
                K,
                50,
                ir.as_mut_ptr(),
                dr.as_mut_ptr(),
            );
            assert_eq!(nc, K, "cpp hnsw_search returned short at query {qi}");
            assert_eq!(nr, K, "rs hnsw_search returned short at query {qi}");
            rec_c += ground_truth::recall_at_k(&ic[..nc], &truth);
            rec_r += ground_truth::recall_at_k(&ir[..nr], &truth);
        }
        let rec_c = rec_c / N_QUERIES as f32;
        let rec_r = rec_r / N_QUERIES as f32;
        let (cpp, rs) = median_pair_ns(
            || {
                ffi::vanedb_cpp_hnsw_search(
                    hc,
                    q.as_ptr(),
                    K,
                    50,
                    ic.as_mut_ptr(),
                    dc.as_mut_ptr(),
                );
            },
            || {
                ffi::vanedb_rs_hnsw_search(hr, q.as_ptr(), K, 50, ir.as_mut_ptr(), dr.as_mut_ptr());
            },
        );
        out.push_str(&format!(
            "| hnsw_search | {cpp} | {rs} | {:.2} |\n",
            rs as f64 / cpp as f64
        ));
        ffi::vanedb_cpp_hnsw_free(hc);
        ffi::vanedb_rs_hnsw_free(hr);
        out.push_str(&format!(
            "\nHNSW recall@{K}: C++ {rec_c:.3}, Rust {rec_r:.3}\n"
        ));
    }

    std::fs::write("RESULTS.md", &out).unwrap();
    print!("{out}");
}
