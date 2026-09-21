//! Automatic beam widening must actually widen at default settings.
//!
//! `max_ef_search` bounds beam width alone. A single `ef_search = 50` pass on
//! an `M = 16` graph already visits well over 4 x 50 nodes, so any exit
//! condition that compares the visit count against the cap ends widening
//! after one pass and a selective filter returns fewer than `k` with no
//! signal -- the situation RFC 0004 rejects "no automatic widening" for.
//! Both tests here fail against that clause.

use std::sync::atomic::{AtomicUsize, Ordering};

use vanedb::approx::{ApproxIndex, Filter, SearchParams};
use vanedb::distance::Metric;

/// SplitMix64: deterministic, dependency-free.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.unit()).collect()
    }
}

const N: usize = 10_000;
const DIM: usize = 16;
const K: usize = 10;
const SEED: u64 = 0x51DE_F1EE;

fn build(rng: &mut Rng) -> ApproxIndex {
    let index = ApproxIndex::builder(DIM, Metric::L2)
        .capacity(N)
        .seed(SEED)
        .build()
        .unwrap();
    let ids: Vec<u64> = (0..N as u64).collect();
    let mut data = Vec::with_capacity(N * DIM);
    for _ in 0..N {
        data.extend(rng.vector(DIM));
    }
    index.add_batch(&ids, &data).unwrap();
    index
}

#[test]
fn one_percent_allowlist_returns_k_at_default_settings() {
    let mut rng = Rng(SEED);
    let index = build(&mut rng);
    // 1% selectivity: 100 of 10,000 ids, strictly ascending.
    let allowed: Vec<u64> = (0..N as u64).filter(|id| id % 100 == 7).collect();
    let params = SearchParams::new().filter(Filter::Allow(&allowed));

    let queries = 100;
    let mut full = 0;
    for _ in 0..queries {
        let query = rng.vector(DIM);
        let hits = index.search_with(&query, K, &params).unwrap();
        assert!(hits.iter().all(|h| allowed.binary_search(&h.id).is_ok()));
        if hits.len() == K {
            full += 1;
        }
    }
    eprintln!("queries returning k={K} at 1%: {full}/{queries}");
    // Measured on this fixture and seed: 100 of 100 with widening, 32 of 100
    // with the visit-count exit. The threshold leaves a margin for a graph
    // built with a different `add_batch` topology.
    assert!(
        full >= 90,
        "only {full}/{queries} queries returned {K} matches at default settings"
    );
}

#[test]
fn reject_all_predicate_widens_past_one_pass_at_default_cap() {
    let mut rng = Rng(SEED);
    let index = build(&mut rng);
    let query = rng.vector(DIM);
    // Pin the initial beam per query rather than reading the index default,
    // so the two searches below differ only in their cap.
    let ef = 50;

    // A reject-all predicate never fills a result slot, so it is called once
    // per scored live node and its call count is the per-pass visit count.
    let calls = AtomicUsize::new(0);
    let reject = |_: u64| {
        calls.fetch_add(1, Ordering::Relaxed);
        false
    };

    let one_pass = SearchParams::new()
        .filter(Filter::Predicate(&reject))
        .ef_search(ef)
        .max_ef_search(ef);
    assert!(index.search_with(&query, K, &one_pass).unwrap().is_empty());
    let single = calls.swap(0, Ordering::Relaxed);

    let default_cap = SearchParams::new()
        .filter(Filter::Predicate(&reject))
        .ef_search(ef);
    assert!(index
        .search_with(&query, K, &default_cap)
        .unwrap()
        .is_empty());
    let widened = calls.load(Ordering::Relaxed);

    // Measured on this fixture and seed: 833 for one pass, 4,504 with the
    // default cap, and 833 again with the visit-count exit.
    eprintln!("single pass visited {single}, default cap visited {widened}");
    // One pass already visits more than 4 x ef nodes; that is exactly why a
    // visit-count exit would stop here.
    assert!(single > 4 * ef, "one pass visited only {single} nodes");
    // The default cap allows ef -> 2ef -> 4ef: three passes, each wider than
    // the last, so the total is well over twice a single pass.
    assert!(
        widened >= 2 * single,
        "default cap visited {widened} nodes, single pass {single}: widening did not run"
    );
}
