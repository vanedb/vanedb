//! RFC 0010 changed the hasher behind every internal id map and the shape of
//! the finite check. Neither may change a saved byte or a result order.
//!
//! Two kinds of anchor:
//!
//! - **The conformance fixtures.** Every VNDB v1 store and every VNDB v2
//!   graph golden is loaded and written back, and the bytes must match the
//!   fixture. The format tests already prove each engine reads and reproduces
//!   the fixtures; this file states the same thing from the write path's
//!   side so a hasher or growth-strategy change that reorders anything trips
//!   here by name.
//! - **A pre-change expectation.** `tests/fixtures/rfc0010/` holds what
//!   `origin/main` at 51cb2d8 produced, before the hasher change, for a fixed
//!   workload: search results and vector lookups from all three index types,
//!   plus the exact bytes `ApproxIndex::to_bytes` and `DiskIndexBuilder::save`
//!   wrote. The ids mix the families the hasher must cope with -- sequential,
//!   low bits shared, and random 64-bit -- and the workload includes removes,
//!   so the swap-remove and tombstone paths are pinned too.
//!
//! There is no public iteration over ids, so result order is the only order
//! a caller can observe; it is decided by distance and id, never by the map.
//!
//! The expectation was captured by running this file on that commit with
//! `VANEDB_RFC0010_CAPTURE=<dir>` set, which writes the fixtures instead of
//! comparing against them. The `ApproxIndex` bytes depend on `StdRng`'s level
//! draws, so a `rand` upgrade that changes the stream needs a re-capture --
//! the same review `conformance/graph/README.md` already asks for.

use std::path::{Path, PathBuf};

use vanedb::{ApproxIndex, FlatIndex, Metric, SearchResult};

#[cfg(feature = "disk")]
use vanedb::{DiskIndex, DiskIndexBuilder};

const DIM: usize = 8;
const N: usize = 300;
const K: usize = 10;
const QUERIES: usize = 5;

fn fixtures(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(sub)
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vanedb_rfc0010_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// SplitMix64, written out so the workload is the same on every platform
/// and every dependency version.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on [-1, 1), quantised to 24 bits so the value is exact.
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

/// Ids drawn from the three families the hasher is tested against, so the
/// expectation pins result order under every one of them at once.
fn ids() -> Vec<u64> {
    let mut rng = Rng(0x1D5);
    let ids: Vec<u64> = (0..N as u64)
        .map(|i| match i % 3 {
            0 => i,
            1 => (i + 1) << 20,
            _ => rng.next_u64() | (1 << 63),
        })
        .collect();
    let distinct: std::collections::HashSet<u64> = ids.iter().copied().collect();
    assert_eq!(distinct.len(), ids.len(), "workload ids must be distinct");
    ids
}

fn vectors() -> Vec<f32> {
    let mut rng = Rng(0xA0010);
    (0..N * DIM).map(|_| rng.unit()).collect()
}

fn queries() -> Vec<Vec<f32>> {
    let mut rng = Rng(0xC0010);
    (0..QUERIES)
        .map(|_| (0..DIM).map(|_| rng.unit()).collect())
        .collect()
}

/// Every seventh id, removed after the adds.
fn removed(ids: &[u64]) -> Vec<u64> {
    ids.iter().copied().step_by(7).collect()
}

/// One line per search: ids in result order with the exact distance bits.
fn line(tag: &str, hits: &[SearchResult]) -> String {
    let body: Vec<String> = hits
        .iter()
        .map(|h| format!("{}:{:08x}", h.id, h.distance.to_bits()))
        .collect();
    format!("{tag} {}\n", body.join(","))
}

fn vector_line(tag: &str, id: u64, vector: &[f32]) -> String {
    let body: Vec<String> = vector
        .iter()
        .map(|v| format!("{:08x}", v.to_bits()))
        .collect();
    format!("{tag} {id} {}\n", body.join(","))
}

struct Observed {
    results: String,
    approx_bytes: Vec<u8>,
    #[cfg(feature = "disk")]
    disk_bytes: Vec<u8>,
}

fn observe() -> Observed {
    let ids = ids();
    let vectors = vectors();
    let queries = queries();
    let removed = removed(&ids);
    let mut results = String::new();

    // FlatIndex: half through `add`, half through `add_batch`, then removes,
    // which swap the tail into the freed slot.
    let flat = FlatIndex::new(DIM, Metric::L2).unwrap();
    let split = N / 2;
    for (i, &id) in ids[..split].iter().enumerate() {
        flat.add(id, &vectors[i * DIM..(i + 1) * DIM]).unwrap();
    }
    flat.add_batch(&ids[split..], &vectors[split * DIM..])
        .unwrap();
    for &id in &removed {
        flat.remove(id).unwrap();
    }
    assert_eq!(flat.len(), N - removed.len());
    for (q, query) in queries.iter().enumerate() {
        results.push_str(&line(
            &format!("flat q{q}"),
            &flat.search(query, K).unwrap(),
        ));
    }
    for &id in ids.iter().skip(1).step_by(50) {
        results.push_str(&vector_line("flat get", id, &flat.get(id).unwrap()));
    }

    // ApproxIndex: seeded, ef_search wide enough that the results are exact
    // for this corpus, so the lines below pin distance-and-id order rather
    // than graph luck. The bytes pin the graph itself.
    let approx = ApproxIndex::builder(DIM, Metric::Cosine)
        .capacity(N)
        .m(8)
        .ef_construction(64)
        .seed(0x0010)
        .build()
        .unwrap();
    approx.set_ef_search(N);
    for (i, &id) in ids[..split].iter().enumerate() {
        approx.add(id, &vectors[i * DIM..(i + 1) * DIM]).unwrap();
    }
    approx
        .add_batch(&ids[split..], &vectors[split * DIM..])
        .unwrap();
    for &id in &removed {
        approx.remove(id).unwrap();
    }
    assert_eq!(approx.len(), N - removed.len());
    for (q, query) in queries.iter().enumerate() {
        results.push_str(&line(
            &format!("approx q{q}"),
            &approx.search(query, K).unwrap(),
        ));
    }
    let approx_bytes = approx.to_bytes().unwrap();

    #[cfg(feature = "disk")]
    let disk_bytes = {
        let mut builder = DiskIndexBuilder::new(DIM, Metric::Dot).unwrap();
        for (i, &id) in ids.iter().enumerate() {
            builder.add(id, &vectors[i * DIM..(i + 1) * DIM]).unwrap();
        }
        let path = scratch("disk").join("workload.vndb");
        builder.save(&path).unwrap();
        // SAFETY: the file was just written by this test and nothing else
        // touches it while it is mapped.
        let disk = unsafe { DiskIndex::open(&path) }.unwrap();
        for (q, query) in queries.iter().enumerate() {
            results.push_str(&line(
                &format!("disk q{q}"),
                &disk.search(query, K).unwrap(),
            ));
        }
        for &id in ids.iter().skip(2).step_by(50) {
            results.push_str(&vector_line("disk get", id, &disk.get(id).unwrap()));
        }
        std::fs::read(&path).unwrap()
    };

    Observed {
        results,
        approx_bytes,
        #[cfg(feature = "disk")]
        disk_bytes,
    }
}

fn compare_bytes(name: &str, expected: &[u8], actual: &[u8]) {
    if expected == actual {
        return;
    }
    let first = expected
        .iter()
        .zip(actual)
        .position(|(a, b)| a != b)
        .unwrap_or(expected.len().min(actual.len()));
    panic!(
        "{name}: {} expected bytes, {} actual, first difference at offset {first}",
        expected.len(),
        actual.len()
    );
}

#[test]
fn workload_matches_the_pre_change_expectation() {
    let observed = observe();
    if let Ok(dir) = std::env::var("VANEDB_RFC0010_CAPTURE") {
        let dir = Path::new(&dir);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("expected_results.txt"), &observed.results).unwrap();
        std::fs::write(dir.join("expected_approx.vndb"), &observed.approx_bytes).unwrap();
        #[cfg(feature = "disk")]
        std::fs::write(dir.join("expected_disk.vndb"), &observed.disk_bytes).unwrap();
        return;
    }

    let expected = include_str!("fixtures/rfc0010/expected_results.txt");
    // The disk lines exist only when the feature is on; the fixture was
    // captured with it, so compare the lines this build can produce.
    let expected: String = expected
        .lines()
        .filter(|l| cfg!(feature = "disk") || !l.starts_with("disk "))
        .map(|l| format!("{l}\n"))
        .collect();
    assert_eq!(
        observed.results, expected,
        "search results or lookups changed"
    );

    compare_bytes(
        "ApproxIndex::to_bytes",
        include_bytes!("fixtures/rfc0010/expected_approx.vndb"),
        &observed.approx_bytes,
    );
    #[cfg(feature = "disk")]
    compare_bytes(
        "DiskIndexBuilder::save",
        include_bytes!("fixtures/rfc0010/expected_disk.vndb"),
        &observed.disk_bytes,
    );
}

/// Every VNDB v2 graph golden survives a load and a write-back unchanged.
#[test]
fn graph_goldens_are_written_back_byte_for_byte() {
    let dir = fixtures("vndb_graph");
    let mut seen = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("vndb") {
            continue;
        }
        seen += 1;
        let golden = std::fs::read(&path).unwrap();
        let index = ApproxIndex::load(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        compare_bytes(&name, &golden, &index.to_bytes().unwrap());
        // The path writer is the same codec behind an atomic rename.
        let out = scratch(&name).join("resaved.vndb");
        index.save(&out).unwrap();
        compare_bytes(
            &format!("{name} via save"),
            &golden,
            &std::fs::read(&out).unwrap(),
        );
    }
    assert_eq!(
        seen, 13,
        "the graph golden set changed size; update the count"
    );
}

/// Every VNDB v1 store golden, rebuilt from what a reader sees, writes the
/// same bytes. `DiskIndex` is read-only, so the write-back goes through the
/// builder in the store's own id order.
#[cfg(feature = "disk")]
#[test]
fn disk_goldens_are_written_back_byte_for_byte() {
    let dir = fixtures("conformance/vndb");
    let mut seen = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("vndb") {
            continue;
        }
        seen += 1;
        let golden = std::fs::read(&path).unwrap();
        // SAFETY: these checked-in fixtures are not modified by the test.
        let index = unsafe { DiskIndex::open(&path) }.unwrap_or_else(|e| panic!("{path:?}: {e}"));
        let mut builder = DiskIndexBuilder::new(index.dimension(), index.metric()).unwrap();
        // Ids are stored in slot order after the 32-byte v1 header (magic,
        // version, dim, count, metric, reserved). The API does not expose the
        // order, so recover it from the file rather than guess it.
        const V1_HEADER: usize = 32;
        let count = index.len();
        for slot in 0..count {
            let off = V1_HEADER + slot * 8;
            let id = u64::from_le_bytes(golden[off..off + 8].try_into().unwrap());
            builder.add(id, &index.get(id).unwrap()).unwrap();
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let out = scratch(&name).join("resaved.vndb");
        builder.save(&out).unwrap();
        compare_bytes(&name, &golden, &std::fs::read(&out).unwrap());
    }
    assert_eq!(
        seen, 3,
        "the store golden set changed size; update the count"
    );
}
