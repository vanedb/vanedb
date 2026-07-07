use criterion::{criterion_group, criterion_main, Criterion};
use std::ffi::CString;
use std::hint::black_box;
use vanedb_bench::{ffi, workloads};

const DIM: usize = 128;
const N: usize = 10_000;

fn bench_mmap_search(c: &mut Criterion) {
    let w = workloads::generate(4, DIM, N, 1);
    let q = &w.queries[0..DIM];
    let cpp_path = CString::new("bench_cpp.mmap").unwrap();
    let rs_path = CString::new("bench_rs.mmap").unwrap();

    unsafe {
        ffi::vanedb_cpp_mmap_build(
            cpp_path.as_ptr(),
            DIM,
            0,
            w.ids.as_ptr(),
            w.vectors.as_ptr(),
            N,
        );
        ffi::vanedb_rs_mmap_build(
            rs_path.as_ptr(),
            DIM,
            0,
            w.ids.as_ptr(),
            w.vectors.as_ptr(),
            N,
        );
        let mc = ffi::vanedb_cpp_mmap_open(cpp_path.as_ptr());
        let mr = ffi::vanedb_rs_mmap_open(rs_path.as_ptr());
        let mut ids = [0u64; 10];
        let mut ds = [0f32; 10];
        let mut g = c.benchmark_group("mmap_search");
        g.bench_function("cpp", |bn| {
            bn.iter(|| {
                ffi::vanedb_cpp_mmap_search(
                    mc,
                    black_box(q.as_ptr()),
                    10,
                    ids.as_mut_ptr(),
                    ds.as_mut_ptr(),
                )
            })
        });
        g.bench_function("rs", |bn| {
            bn.iter(|| {
                ffi::vanedb_rs_mmap_search(
                    mr,
                    black_box(q.as_ptr()),
                    10,
                    ids.as_mut_ptr(),
                    ds.as_mut_ptr(),
                )
            })
        });
        g.finish();
        ffi::vanedb_cpp_mmap_free(mc);
        ffi::vanedb_rs_mmap_free(mr);
    }
    let _ = std::fs::remove_file("bench_cpp.mmap");
    let _ = std::fs::remove_file("bench_rs.mmap");
}

criterion_group!(benches, bench_mmap_search);
criterion_main!(benches);
