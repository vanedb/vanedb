//! Requested live heap bytes only: no timings, RSS, or device-capacity claims.
use sha2::{Digest, Sha256};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, DefaultHasher};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use vanedb::{ApproxIndex, DiskIndex, DiskIndexBuilder, FlatIndex, Metric};

const FIXTURE_SHA: &str = "4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d";
const SOURCE_SHA: &str = "35758c6ccac0b9e657be8d80e84d1744378df120";
const DIM: usize = 768;

struct Counting(AtomicUsize);
impl Counting {
    const fn new() -> Self {
        Self(AtomicUsize::new(0))
    }
    fn live(&self) -> usize {
        self.0.load(SeqCst)
    }
    // The pointer result controls accounting: failed alloc/realloc changes nothing.
    fn allocated(&self, p: *mut u8, size: usize) {
        if !p.is_null() {
            self.0.fetch_add(size, SeqCst);
        }
    }
    fn resized(&self, p: *mut u8, old: usize, new: usize) {
        if !p.is_null() {
            if new >= old {
                self.0.fetch_add(new - old, SeqCst);
            } else {
                self.0.fetch_sub(old - new, SeqCst);
            }
        }
    }
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        self.allocated(p, l.size());
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        self.allocated(p, l.size());
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        self.0.fetch_sub(l.size(), SeqCst);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, size: usize) -> *mut u8 {
        let result = System.realloc(p, l, size);
        self.resized(result, l.size(), size);
        result
    }
}
#[global_allocator]
static HEAP: Counting = Counting::new();

fn self_test() {
    // Safe containers exercise the global allocator without raw-pointer access.
    // black_box keeps the allocations observable in this optimized executable.
    let baseline = HEAP.live();
    let mut bytes = Vec::<u8>::new();
    bytes.reserve_exact(std::hint::black_box(113));
    bytes.resize(113, 31);
    std::hint::black_box(&mut bytes);
    assert_eq!(bytes.capacity(), 113);
    assert_eq!(HEAP.live() - baseline, 113);

    bytes.reserve_exact(std::hint::black_box(257 - bytes.len()));
    std::hint::black_box(&mut bytes);
    assert_eq!(bytes.capacity(), 257);
    assert_eq!(bytes[0], 31);
    assert_eq!(HEAP.live() - baseline, 257);

    bytes.truncate(17);
    bytes.shrink_to_fit();
    std::hint::black_box(&mut bytes);
    assert_eq!(bytes.capacity(), 17);
    assert!(bytes.iter().all(|byte| *byte == 31));
    assert_eq!(HEAP.live() - baseline, 17);
    drop(bytes);
    assert_eq!(HEAP.live(), baseline);

    let zeros = vec![0u8; std::hint::black_box(113)];
    std::hint::black_box(&zeros);
    assert!(zeros.iter().all(|byte| *byte == 0));
    assert_eq!(HEAP.live() - baseline, 113);
    drop(zeros);
    assert_eq!(HEAP.live(), baseline);

    // Simulate failed return values without an OOM request or dereferencing them.
    let c = Counting(AtomicUsize::new(113));
    c.allocated(std::ptr::null_mut(), 999);
    c.resized(std::ptr::null_mut(), 113, 999);
    assert_eq!(c.live(), 113);
    println!("allocator calibration PASS: alloc/zeroed/grow/shrink/dealloc/null results");
}

fn fixture(path: &Path, n: usize) -> (Vec<u64>, Vec<f32>) {
    let bytes = std::fs::read(path).expect("read fixture");
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), FIXTURE_SHA);
    let u32_at = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
    assert_eq!(&bytes[..4], b"VNEF");
    assert_eq!(u32_at(4), 1);
    assert_eq!(u32_at(8), DIM);
    assert_eq!(u32_at(12), 100_000);
    assert_eq!(u32_at(16), 1000);
    assert_eq!(u32_at(20), 1);
    assert_eq!(u32_at(24), 0);
    let id_offset = 28 + (100_000 + 1000) * DIM * 4;
    assert_eq!(bytes.len(), id_offset + 100_000 * 8);
    assert!(n > 0 && n <= 100_000);
    assert!(bytes[28..id_offset]
        .chunks_exact(4)
        .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()));
    let all_ids: Vec<u64> = bytes[id_offset..]
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(
        all_ids.iter().copied().collect::<HashSet<_>>().len(),
        100_000
    );
    let ids = all_ids[..n].to_vec();
    let vectors = bytes[28..28 + n * DIM * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    (ids, vectors)
}

fn signed(a: usize, b: usize) -> i128 {
    a as i128 - b as i128
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--self-test") {
        self_test();
        return;
    }
    assert_eq!(
        args.len(),
        5,
        "usage: probe flat|approx|disk N FIXTURE OUTFILE"
    );
    let kind = args[1].as_str();
    let n: usize = args[2].parse().unwrap();
    let path = Path::new(&args[4]);
    assert!(!path.exists(), "refuse to overwrite retained output");
    let (ids, vectors) = fixture(Path::new(&args[3]), n);
    // Separate standard-library map calibration: the control-group tail is
    // target dependent (the historical Linux x86-64 study used 16 bytes).
    // Stateless hasher avoids RandomState initialization; it affects bucket
    // placement but not the allocation layout of (u64, usize) entries.
    let map_baseline = HEAP.live();
    let mut calibration_map: HashMap<u64, usize, BuildHasherDefault<DefaultHasher>> =
        HashMap::with_capacity_and_hasher(n, Default::default());
    for (slot, id) in ids.iter().enumerate() {
        calibration_map.insert(*id, slot);
    }
    let map_live = HEAP.live();
    let map_capacity = calibration_map.capacity();
    drop(calibration_map);
    assert_eq!(
        HEAP.live(),
        map_baseline,
        "calibration map leaked allocations"
    );
    let map_requested_bytes = map_live - map_baseline;
    // All input and path/argument storage are held constant across snapshots.
    // Disk builder is intentionally gone before opening the mapped index.
    if kind == "disk" {
        let mut builder = DiskIndexBuilder::new(DIM, Metric::Cosine).unwrap();
        for (id, v) in ids.iter().zip(vectors.chunks_exact(DIM)) {
            builder.add(*id, v).unwrap();
        }
        builder.save(path).unwrap();
        drop(builder);
    }
    let baseline = HEAP.live();
    let (empty, built, saved, dropped, count) = match kind {
        "flat" => {
            let idx = FlatIndex::new(DIM, Metric::Cosine).unwrap();
            let empty = HEAP.live();
            idx.add_batch(&ids, &vectors).unwrap();
            let built = HEAP.live();
            let count = idx.len();
            drop(idx);
            (empty, built, built, HEAP.live(), count)
        }
        "approx" => {
            let idx = ApproxIndex::builder(DIM, Metric::Cosine)
                .capacity(n)
                .m(16)
                .ef_construction(200)
                .seed(7)
                .build()
                .unwrap();
            idx.set_ef_search(50);
            let empty = HEAP.live();
            idx.add_batch(&ids, &vectors).unwrap();
            let built = HEAP.live();
            let count = idx.len();
            idx.save(path).unwrap();
            let saved = HEAP.live();
            drop(idx);
            (empty, built, saved, HEAP.live(), count)
        }
        "disk" => {
            // File is owned solely by this fresh process, immutable while mapped.
            let idx = unsafe { DiskIndex::open(path) }.unwrap();
            let built = HEAP.live();
            let count = idx.len();
            drop(idx);
            (baseline, built, built, HEAP.live(), count)
        }
        _ => panic!("unknown index kind"),
    };
    assert_eq!(count, n);
    // If persistence leaves allocations, do not silently attribute them to the index.
    assert_eq!(
        saved, built,
        "save changed retained requested heap; investigate"
    );
    assert!(built >= dropped);
    let output = serde_json::json!({
        "kind": kind, "n": n, "dim": DIM, "metric": "cosine", "pid": std::process::id(),
        "source_sha": SOURCE_SHA, "fixture_sha256": FIXTURE_SHA,
        "fixture_selection": "first n document rows in fixture order, original IDs; no query inserts",
        "capacity_hint": if kind == "approx" { Some(n) } else { None },
        "m": if kind == "approx" { Some(16) } else { None },
        "ef_construction": if kind == "approx" { Some(200) } else { None },
        "ef_search": if kind == "approx" { Some(50) } else { None },
        "seed": if kind == "approx" { Some(7) } else { None },
        "baseline_live_bytes": baseline, "empty_live_bytes": empty,
        "built_live_bytes": built, "after_save_live_bytes": saved, "after_drop_live_bytes": dropped,
        "empty_delta_bytes": signed(empty, baseline), "built_delta_bytes": signed(built, baseline),
        "retained_after_drop_delta_bytes": signed(dropped, baseline),
        "released_on_drop_bytes": signed(built, dropped),
        "usize_bytes": std::mem::size_of::<usize>(), "vec_header_bytes": std::mem::size_of::<Vec<usize>>(),
        "file": if kind == "flat" { None } else { Some(path) },
        "file_bytes": if kind == "flat" { None } else { Some(std::fs::metadata(path).unwrap().len()) },
        "calibration_map_requested_bytes": map_requested_bytes,
        "calibration_map_usable_capacity": map_capacity,
        "calibration_map_residual_bytes": 0,
        "scope": "requested heap bytes; graph built delta includes retained thread-local scratch; no allocator overhead, stack, mappings, RSS, or timings"
    });
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
    // Keep input arrays alive until after every heap snapshot.
    std::hint::black_box((&ids, &vectors));
}
