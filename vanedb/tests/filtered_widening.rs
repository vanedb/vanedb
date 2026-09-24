//! Automatic beam widening must actually widen at default settings.
//!
//! `max_ef_search` bounds beam width alone. A single `ef_search = 50` pass on
//! an `M = 16` graph already visits well over 4 x 50 nodes, so any exit
//! condition that compares the visit count against the cap ends widening
//! after one pass and a selective filter returns fewer than `k` with no
//! signal -- the situation RFC 0004 rejects "no automatic widening" for.
//! The two 10k tests here fail against that clause; the last test pins the
//! one early exit that remains, a pass that scored every stored slot.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

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

/// The 10k index, built once for the whole binary (a debug build takes
/// ~20 s), and a generator positioned just after that build so each test
/// draws the same queries it drew when it built its own copy.
fn shared() -> (&'static ApproxIndex, Rng) {
    static SHARED: OnceLock<(ApproxIndex, u64)> = OnceLock::new();
    let (index, state) = SHARED.get_or_init(|| {
        let mut rng = Rng(SEED);
        let index = build(&mut rng);
        (index, rng.0)
    });
    (index, Rng(*state))
}

#[test]
fn one_percent_allowlist_returns_k_at_default_settings() {
    let (index, mut rng) = shared();
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
    let (index, mut rng) = shared();
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

    // An explicit, non-power-of-two cap must also retry even when the
    // first pass already scored more nodes than that beam ceiling.
    let explicit_cap = SearchParams::new()
        .filter(Filter::Predicate(&reject))
        .ef_search(ef)
        .max_ef_search(75);
    assert!(index
        .search_with(&query, K, &explicit_cap)
        .unwrap()
        .is_empty());
    let explicit = calls.swap(0, Ordering::Relaxed);
    assert!(single > 75);
    assert!(explicit > single, "explicit cap must permit a second pass");

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

/// The remaining early exit: a pass that scored every stored slot ends
/// widening, because a wider beam cannot reach anything new. On a complete
/// graph (`N <= M`) expanding the entry point marks every slot, so the
/// first pass scores everything. A reject-all predicate is called once per
/// scored live node per pass, so its call count equals the live count and
/// any extra pass shows up as a multiple of it.
#[test]
fn a_pass_that_scored_every_slot_ends_widening() {
    const SLOTS: u64 = 8;
    let index = ApproxIndex::builder(2, Metric::L2)
        .capacity(SLOTS as usize)
        .m(16)
        .build()
        .unwrap();
    for id in 0..SLOTS {
        index.add(id, &[id as f32, (id * 3 % 5) as f32]).unwrap();
    }
    // A tombstone is still a stored slot: it is traversed and counted, but
    // the predicate never sees it.
    index.remove(3).unwrap();
    let live = SLOTS as usize - 1;

    let calls = AtomicUsize::new(0);
    let reject = |_: u64| {
        calls.fetch_add(1, Ordering::Relaxed);
        false
    };
    // Base beam 1, cap clamped to the 8 stored slots. Without the exit the
    // loop runs passes at ef 1, 2, 4 and 8 and calls the predicate 4 x 7
    // times; with it the first pass, having scored all 8 slots, is the last.
    let params = SearchParams::new()
        .filter(Filter::Predicate(&reject))
        .ef_search(1)
        .max_ef_search(64);
    assert!(index
        .search_with(&[0.0, 0.0], 1, &params)
        .unwrap()
        .is_empty());
    let n = calls.load(Ordering::Relaxed);
    eprintln!("predicate calls on a complete graph: {n}");
    assert_eq!(
        n, live,
        "the first pass scored every slot; widening must stop there"
    );
}
