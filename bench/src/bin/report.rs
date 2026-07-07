use std::hint::black_box;
use std::time::Instant;
use vanedb_bench::{ffi, ground_truth, workloads};

const DIM: usize = 128;
const N: usize = 10_000;
const K: usize = 10;

fn median_ns(mut f: impl FnMut()) -> u128 {
    let mut samples: Vec<u128> = (0..50)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos()
        })
        .collect();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

const N_QUERIES: usize = 100;

fn main() {
    let w = workloads::generate(99, DIM, N, N_QUERIES);
    let q = &w.queries[0..DIM];
    let mut out = String::from("# VaneDB Benchmark Results\n\n");
    out.push_str(&format!(
        "Workload: dim={DIM}, n={N}, k={K}, L2. Latencies are medians (one query); \
         recall is averaged over {N_QUERIES} queries.\n\n"
    ));
    out.push_str("| Op | C++ (ns) | Rust (ns) | ratio (rs/cpp) |\n|---|---:|---:|---:|\n");

    unsafe {
        // Distance — timed in batches of 1000: a single ~10 ns call is below
        // Instant::now() resolution.
        let a = &w.vectors[0..DIM];
        let b = &w.vectors[DIM..2 * DIM];
        let cpp = median_ns(|| {
            for _ in 0..1000 {
                black_box(ffi::vanedb_cpp_l2_sq(a.as_ptr(), b.as_ptr(), DIM));
            }
        }) / 1000;
        let rs = median_ns(|| {
            for _ in 0..1000 {
                black_box(ffi::vanedb_rs_l2_sq(a.as_ptr(), b.as_ptr(), DIM));
            }
        }) / 1000;
        out.push_str(&format!(
            "| l2_sq | {cpp} | {rs} | {:.2} |\n",
            rs as f64 / cpp as f64
        ));

        // Store search
        let sc = ffi::vanedb_cpp_store_new(DIM, 0);
        let sr = ffi::vanedb_rs_store_new(DIM, 0);
        for i in 0..N {
            ffi::vanedb_cpp_store_add(sc, w.ids[i], w.vectors[i * DIM..].as_ptr());
            ffi::vanedb_rs_store_add(sr, w.ids[i], w.vectors[i * DIM..].as_ptr());
        }
        let mut ids = [0u64; K];
        let mut ds = [0f32; K];
        let cpp = median_ns(|| {
            ffi::vanedb_cpp_store_search(sc, q.as_ptr(), K, ids.as_mut_ptr(), ds.as_mut_ptr());
        });
        let rs = median_ns(|| {
            ffi::vanedb_rs_store_search(sr, q.as_ptr(), K, ids.as_mut_ptr(), ds.as_mut_ptr());
        });
        out.push_str(&format!(
            "| store_search | {cpp} | {rs} | {:.2} |\n",
            rs as f64 / cpp as f64
        ));
        ffi::vanedb_cpp_store_free(sc);
        ffi::vanedb_rs_store_free(sr);

        // HNSW search + recall@k vs brute-force truth, averaged over all queries
        let hc = ffi::vanedb_cpp_hnsw_new(DIM, 0, N, 16, 200, 7);
        let hr = ffi::vanedb_rs_hnsw_new(DIM, 0, N, 16, 200, 7);
        for i in 0..N {
            ffi::vanedb_cpp_hnsw_add(hc, w.ids[i], w.vectors[i * DIM..].as_ptr());
            ffi::vanedb_rs_hnsw_add(hr, w.ids[i], w.vectors[i * DIM..].as_ptr());
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
            rec_c += ground_truth::recall_at_k(&ic[..nc], &truth);
            rec_r += ground_truth::recall_at_k(&ir[..nr], &truth);
        }
        let rec_c = rec_c / N_QUERIES as f32;
        let rec_r = rec_r / N_QUERIES as f32;
        let cpp = median_ns(|| {
            ffi::vanedb_cpp_hnsw_search(hc, q.as_ptr(), K, 50, ic.as_mut_ptr(), dc.as_mut_ptr());
        });
        let rs = median_ns(|| {
            ffi::vanedb_rs_hnsw_search(hr, q.as_ptr(), K, 50, ir.as_mut_ptr(), dr.as_mut_ptr());
        });
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
