use criterion::{criterion_group, criterion_main, Criterion};
use std::ffi::CString;
use std::hint::black_box;
use std::os::raw::c_char;
use std::time::{Duration, Instant};
use vanedb_bench::coverage::groups;
use vanedb_bench::{ffi, workloads};

const DIM: usize = 128;
const N: usize = 10_000;

// Files are created in the working directory on purpose. The documented
// invocation runs from the repository root, on a real filesystem; /tmp is
// tmpfs on many Linux distributions, which would benchmark RAM rather than
// the mapped-file path these stores exist for.
const CPP_FILE: &str = "bench_cpp.mmap";
const RS_FILE: &str = "bench_rs.mmap";
const CPP_BUILD_FILE: &str = "bench_cpp_build.mmap";
const RS_BUILD_FILE: &str = "bench_rs_build.mmap";

/// Both engines' `mmap_build` share this signature. The indirect call costs
/// nanoseconds against a multi-megabyte write.
type MmapBuildFn =
    unsafe extern "C" fn(*const c_char, usize, u32, *const u64, *const f32, usize) -> i32;

/// # Safety
/// `build` must be one of the two engines' `mmap_build` entry points.
unsafe fn build_into(build: MmapBuildFn, path: &CString, w: &workloads::Workload) -> i32 {
    build(path.as_ptr(), DIM, 0, w.ids.as_ptr(), w.vectors.as_ptr(), N)
}

fn bench_mmap(c: &mut Criterion) {
    let w = workloads::generate(4, DIM, N, 1);
    let q = &w.queries[0..DIM];
    let cpp_path = CString::new(CPP_FILE).unwrap();
    let rs_path = CString::new(RS_FILE).unwrap();
    let cpp_build_path = CString::new(CPP_BUILD_FILE).unwrap();
    let rs_build_path = CString::new(RS_BUILD_FILE).unwrap();

    unsafe {
        // Build the stores the open and search groups read. A failed
        // build/open (e.g. read-only cwd) must fail loudly, not benchmark a
        // null handle as infinitely fast.
        assert_eq!(
            build_into(ffi::vanedb_cpp_mmap_build, &cpp_path, &w),
            0,
            "cpp mmap_build failed"
        );
        assert_eq!(
            build_into(ffi::vanedb_rs_mmap_build, &rs_path, &w),
            0,
            "rs mmap_build failed"
        );

        // --- build ---------------------------------------------------------
        // Each iteration writes a fresh file: overwriting an existing one
        // takes a different path through the filesystem. Removal is untimed.
        // This group writes ~5 MB and syncs, so its run-to-run spread is far
        // wider than the compute benches — see the README's noise floor.
        let mut build = c.benchmark_group(groups::MMAP_BUILD);
        build.sample_size(10);
        build.bench_function("cpp", |bn| {
            bn.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let _ = std::fs::remove_file(CPP_BUILD_FILE);
                    let start = Instant::now();
                    let rc = build_into(ffi::vanedb_cpp_mmap_build, &cpp_build_path, &w);
                    elapsed += start.elapsed();
                    assert_eq!(rc, 0, "cpp mmap_build failed");
                }
                elapsed
            })
        });
        build.bench_function("rs", |bn| {
            bn.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let _ = std::fs::remove_file(RS_BUILD_FILE);
                    let start = Instant::now();
                    let rc = build_into(ffi::vanedb_rs_mmap_build, &rs_build_path, &w);
                    elapsed += start.elapsed();
                    assert_eq!(rc, 0, "rs mmap_build failed");
                }
                elapsed
            })
        });
        build.finish();

        // --- open ----------------------------------------------------------
        // Warm-cache open: the same file is mapped repeatedly, so this is the
        // mapping and header-validation cost, not first-read I/O. mmap_free is
        // outside the measured interval.
        let mut open = c.benchmark_group(groups::MMAP_OPEN);
        open.bench_function("cpp", |bn| {
            bn.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let start = Instant::now();
                    let m = ffi::vanedb_cpp_mmap_open(cpp_path.as_ptr());
                    elapsed += start.elapsed();
                    assert!(!m.is_null(), "cpp mmap_open failed");
                    ffi::vanedb_cpp_mmap_free(black_box(m));
                }
                elapsed
            })
        });
        open.bench_function("rs", |bn| {
            bn.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let start = Instant::now();
                    let m = ffi::vanedb_rs_mmap_open(rs_path.as_ptr());
                    elapsed += start.elapsed();
                    assert!(!m.is_null(), "rs mmap_open failed");
                    ffi::vanedb_rs_mmap_free(black_box(m));
                }
                elapsed
            })
        });
        open.finish();

        // --- search --------------------------------------------------------
        let mc = ffi::vanedb_cpp_mmap_open(cpp_path.as_ptr());
        let mr = ffi::vanedb_rs_mmap_open(rs_path.as_ptr());
        assert!(!mc.is_null() && !mr.is_null(), "mmap_open failed");
        let mut ids = [0u64; 10];
        let mut ds = [0f32; 10];
        // Warmup outside the timed loops doubles as a liveness check.
        assert_eq!(
            ffi::vanedb_cpp_mmap_search(mc, q.as_ptr(), 10, ids.as_mut_ptr(), ds.as_mut_ptr()),
            10
        );
        assert_eq!(
            ffi::vanedb_rs_mmap_search(mr, q.as_ptr(), 10, ids.as_mut_ptr(), ds.as_mut_ptr()),
            10
        );
        let mut g = c.benchmark_group(groups::MMAP_SEARCH);
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

    for file in [CPP_FILE, RS_FILE, CPP_BUILD_FILE, RS_BUILD_FILE] {
        let _ = std::fs::remove_file(file);
    }
}

criterion_group!(benches, bench_mmap);
criterion_main!(benches);
