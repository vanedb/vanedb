use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;
use vanedb_bench::config::ReportConfig;
use vanedb_bench::{ffi, ground_truth, workloads};

/// Sanity floor, not a performance assertion: a healthy graph recalls ~0.7 at
/// these settings and a broken one collapses to near zero. It exists so the
/// smoke run fails on a graph that builds but does not retrieve.
const MIN_RECALL: f32 = 0.5;

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

fn main() -> ExitCode {
    let cfg = match ReportConfig::from_env() {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("invalid configuration: {err}");
            return ExitCode::FAILURE;
        }
    };
    let ReportConfig {
        dim,
        n,
        k,
        queries,
        out,
    } = cfg;

    let w = workloads::generate(99, dim, n, queries);
    let q = &w.queries[0..dim];
    let mut md = String::from("# VaneDB Benchmark Results\n\n");
    md.push_str(&format!(
        "Engines: vanedb-cpp (CMake Release) and vanedb (Rust), monorepo {}.\n",
        env!("VANEDB_MONOREPO_REV"),
    ));
    md.push_str(&format!(
        "Workload: dim={dim}, n={n}, k={k}, L2. Latencies are medians of 501 \
         interleaved paired samples (one query) after a joint warmup; recall is \
         averaged over {queries} queries. Both engines' data stays resident \
         in one process (interleaved construction).\n\n"
    ));
    // Regenerating this file must not lose the caveats: anything a reader
    // needs alongside the numbers belongs here, not hand-added afterwards.
    md.push_str(&format!(
        "Covers l2_sq, store_search, and index_search + recall@{k} only; every \
         other operation is criterion-only (see README).\n\n\
         Criterion is canonical; see the README table. This bin times l2_sq in \
         batches of 1000 calls, which inlines differently from criterion's \
         per-call harness.\n\n"
    ));
    md.push_str("| Op | C++ (ns) | Rust (ns) | ratio (rs/cpp) |\n|---|---:|---:|---:|\n");

    unsafe {
        // Distance — timed in batches of 1000: a single ~10 ns call is below
        // Instant::now() resolution.
        let a = &w.vectors[0..dim];
        let b = &w.vectors[dim..2 * dim];
        let (cpp, rs) = median_pair_ns(
            || {
                for _ in 0..1000 {
                    black_box(ffi::vanedb_cpp_l2_sq(a.as_ptr(), b.as_ptr(), dim));
                }
            },
            || {
                for _ in 0..1000 {
                    black_box(ffi::vanedb_rs_l2_sq(a.as_ptr(), b.as_ptr(), dim));
                }
            },
        );
        // Ratio from the raw batch totals — dividing to per-call ns first
        // truncates ~13.9 vs ~15.0 into 13 vs 15 and distorts the ratio.
        let ratio = rs as f64 / cpp as f64;
        let (cpp, rs) = (cpp / 1000, rs / 1000);
        md.push_str(&format!("| l2_sq | {cpp} | {rs} | {ratio:.2} |\n"));

        // Store search. Setup asserts keep a failed engine from benchmarking
        // as infinitely fast.
        let sc = ffi::vanedb_cpp_store_new(dim, 0);
        let sr = ffi::vanedb_rs_store_new(dim, 0);
        assert!(!sc.is_null() && !sr.is_null(), "store_new failed");
        for i in 0..n {
            assert_eq!(
                ffi::vanedb_cpp_store_add(sc, w.ids[i], w.vectors[i * dim..].as_ptr()),
                0
            );
            assert_eq!(
                ffi::vanedb_rs_store_add(sr, w.ids[i], w.vectors[i * dim..].as_ptr()),
                0
            );
        }
        let mut ids_c = vec![0u64; k];
        let mut ds_c = vec![0f32; k];
        let mut ids_r = vec![0u64; k];
        let mut ds_r = vec![0f32; k];
        assert_eq!(
            ffi::vanedb_cpp_store_search(sc, q.as_ptr(), k, ids_c.as_mut_ptr(), ds_c.as_mut_ptr()),
            k
        );
        assert_eq!(
            ffi::vanedb_rs_store_search(sr, q.as_ptr(), k, ids_r.as_mut_ptr(), ds_r.as_mut_ptr()),
            k
        );
        let (cpp, rs) = median_pair_ns(
            || {
                ffi::vanedb_cpp_store_search(
                    sc,
                    q.as_ptr(),
                    k,
                    ids_c.as_mut_ptr(),
                    ds_c.as_mut_ptr(),
                );
            },
            || {
                ffi::vanedb_rs_store_search(
                    sr,
                    q.as_ptr(),
                    k,
                    ids_r.as_mut_ptr(),
                    ds_r.as_mut_ptr(),
                );
            },
        );
        md.push_str(&format!(
            "| store_search | {cpp} | {rs} | {:.2} |\n",
            rs as f64 / cpp as f64
        ));
        ffi::vanedb_cpp_store_free(sc);
        ffi::vanedb_rs_store_free(sr);

        // HNSW search + recall@k vs brute-force truth, averaged over all queries
        let hc = ffi::vanedb_cpp_index_new(dim, 0, n, 16, 200, 7);
        let hr = ffi::vanedb_rs_index_new(dim, 0, n, 16, 200, 7);
        assert!(!hc.is_null() && !hr.is_null(), "hnsw_new failed");
        for i in 0..n {
            assert_eq!(
                ffi::vanedb_cpp_index_add(hc, w.ids[i], w.vectors[i * dim..].as_ptr()),
                0
            );
            assert_eq!(
                ffi::vanedb_rs_index_add(hr, w.ids[i], w.vectors[i * dim..].as_ptr()),
                0
            );
        }
        let mut ic = vec![0u64; k];
        let mut dc = vec![0f32; k];
        let mut ir = vec![0u64; k];
        let mut dr = vec![0f32; k];
        let (mut rec_c, mut rec_r) = (0.0f32, 0.0f32);
        for qi in 0..queries {
            let query = &w.queries[qi * dim..(qi + 1) * dim];
            let truth = ground_truth::brute_force_topk(&w.vectors, &w.ids, dim, query, k);
            let nc = ffi::vanedb_cpp_index_search(
                hc,
                query.as_ptr(),
                k,
                50,
                ic.as_mut_ptr(),
                dc.as_mut_ptr(),
            );
            let nr = ffi::vanedb_rs_index_search(
                hr,
                query.as_ptr(),
                k,
                50,
                ir.as_mut_ptr(),
                dr.as_mut_ptr(),
            );
            assert_eq!(nc, k, "cpp index_search returned short at query {qi}");
            assert_eq!(nr, k, "rs index_search returned short at query {qi}");
            rec_c += ground_truth::recall_at_k(&ic[..nc], &truth);
            rec_r += ground_truth::recall_at_k(&ir[..nr], &truth);
        }
        let rec_c = rec_c / queries as f32;
        let rec_r = rec_r / queries as f32;
        let (cpp, rs) = median_pair_ns(
            || {
                ffi::vanedb_cpp_index_search(
                    hc,
                    q.as_ptr(),
                    k,
                    50,
                    ic.as_mut_ptr(),
                    dc.as_mut_ptr(),
                );
            },
            || {
                ffi::vanedb_rs_index_search(hr, q.as_ptr(), k, 50, ir.as_mut_ptr(), dr.as_mut_ptr());
            },
        );
        md.push_str(&format!(
            "| index_search | {cpp} | {rs} | {:.2} |\n",
            rs as f64 / cpp as f64
        ));
        ffi::vanedb_cpp_index_free(hc);
        ffi::vanedb_rs_index_free(hr);
        md.push_str(&format!(
            "\nIndex recall@{k}: C++ {rec_c:.3}, Rust {rec_r:.3}\n"
        ));

        assert!(
            rec_c >= MIN_RECALL && rec_r >= MIN_RECALL,
            "recall@{k} below the {MIN_RECALL} sanity floor: C++ {rec_c:.3}, Rust {rec_r:.3}"
        );
    }

    std::fs::write(&out, &md).unwrap_or_else(|e| panic!("writing {}: {e}", out.display()));
    print!("{md}");
    eprintln!("written to {}", out.display());
    ExitCode::SUCCESS
}
