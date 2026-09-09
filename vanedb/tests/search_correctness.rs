//! Does it actually search? Ranking checked against an independent reference.
//!
//! Every other cross-check in this crate uses `FlatIndex` as ground truth:
//! `disk_tests::mmap_matches_brute_force` compares `DiskIndex` to it, and
//! `approx_tests::hnsw_recall_vs_brute_force` measures recall against it. That
//! makes `FlatIndex` the one index nothing checks — a systematic ordering or
//! distance error in the exact scan would satisfy every one of those tests,
//! because they would all be comparing the engine to itself. `bench/`'s
//! `brute_force_topk` is no help either: it is a second copy of the same f32
//! kernels, in a separate workspace, so it shares the conventions under test.
//!
//! The reference here is computed in this file, in `f64`, from the metric
//! definitions rather than from `vanedb::distance`. It shares no code with the
//! engine, so agreement means the two independently arrive at the same answer.
//!
//! Dimensions are chosen to straddle the SIMD boundaries: AVX2 consumes eight
//! floats per iteration and NEON four, so 7/8/9 and 15/16/17 exercise a full
//! body, an exact fit, and a one-element tail on both.

use vanedb::{ApproxIndex, FlatIndex, Metric, SearchParams};

#[cfg(feature = "disk")]
use vanedb::{DiskIndex, DiskIndexBuilder};

/// SplitMix64. A generator written out here rather than pulled from `rand`:
/// the vectors must be identical on every platform and every dependency
/// version, or a failure is not reproducible from the test name alone.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Roughly uniform on [-1, 1), quantised to 24 bits so the value is exact
    /// in both `f32` and `f64` and the reference loses nothing in conversion.
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / 4_194_304.0 - 1.0
    }

    fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.unit()).collect()
    }
}

/// The metric definitions, in `f64`, written from the specification rather
/// than borrowed from the engine.
///
/// The zero/degenerate policy matches `conformance/README.md`: a vector whose
/// computed squared norm is not a usable positive number is 1.0 from
/// everything. `f64` reaches that state at different magnitudes than `f32`, so
/// the callers below keep their inputs well inside the ordinary range.
fn reference(metric: Metric, a: &[f32], b: &[f32]) -> f64 {
    let (x, y): (Vec<f64>, Vec<f64>) = (
        a.iter().map(|v| *v as f64).collect(),
        b.iter().map(|v| *v as f64).collect(),
    );
    match metric {
        Metric::L2 => x.iter().zip(&y).map(|(p, q)| (p - q) * (p - q)).sum(),
        Metric::Dot => -x.iter().zip(&y).map(|(p, q)| p * q).sum::<f64>(),
        Metric::Cosine => {
            let dot: f64 = x.iter().zip(&y).map(|(p, q)| p * q).sum();
            let na: f64 = x.iter().map(|p| p * p).sum::<f64>().sqrt();
            let nb: f64 = y.iter().map(|q| q * q).sum::<f64>().sqrt();
            let denom = na * nb;
            if !(denom > 0.0 && denom.is_finite()) {
                return 1.0;
            }
            1.0 - (dot / denom).clamp(-1.0, 1.0)
        }
        // Metric is #[non_exhaustive]; a new one needs its definition here, not
        // a silent fallback that would rank against the wrong metric.
        other => panic!("no reference implementation for {other:?}"),
    }
}

/// Reference ranking: ascending distance, ties broken by ascending id — the
/// order `SearchResult`'s `Ord` documents.
fn reference_ranking(
    metric: Metric,
    ids: &[u64],
    vectors: &[Vec<f32>],
    query: &[f32],
) -> Vec<(u64, f64)> {
    let mut scored: Vec<(u64, f64)> = ids
        .iter()
        .zip(vectors)
        .map(|(&id, v)| (id, reference(metric, v, query)))
        .collect();
    scored.sort_by(|(ai, ad), (bi, bd)| ad.total_cmp(bd).then(ai.cmp(bi)));
    scored
}

/// How far down the reference ranking `f32` can be held to the exact order.
///
/// Two vectors whose true distances differ by less than the engine's `f32`
/// arithmetic can resolve may come back in either order without anything being
/// wrong. So the exact-order assertions apply to the leading run of positions
/// the reference separates decisively, and stop at the first pair it does not.
/// This is a real limit rather than a convenience: over the corpora below,
/// ranks 10 and 11 of a 200-vector cosine query landed 4e-8 apart.
///
/// Returns the length of that run, capped at `k`. Callers assert the prefix
/// matches id for id, and separately that the full top-`k` is the right *set*.
fn decisive_prefix(ranked: &[(u64, f64)], k: usize) -> usize {
    let separated = |lo: f64, hi: f64| {
        let scale = lo.abs().max(hi.abs()).max(1.0);
        (hi - lo) / scale > 1e-5
    };
    let limit = k.min(ranked.len());
    for p in 1..limit {
        if !separated(ranked[p - 1].1, ranked[p].1) {
            return p - 1;
        }
    }
    // The k/k+1 boundary decides whether the set itself is well defined.
    if limit < ranked.len() && !separated(ranked[limit - 1].1, ranked[limit].1) {
        return limit - 1;
    }
    limit
}

const DIMS: [usize; 12] = [1, 2, 3, 7, 8, 9, 15, 16, 17, 32, 64, 128];
const METRICS: [Metric; 3] = [Metric::L2, Metric::Cosine, Metric::Dot];

/// The exact index returns the reference ranking, id for id.
#[test]
fn flat_search_reproduces_an_independent_f64_ranking() {
    let mut decided = 0usize;
    // At a realistic `k` the reference almost always separates every rank, so
    // the whole requested top-k is asserted. Tracked because a change that made
    // the reference mushy would otherwise weaken this test in silence. (A
    // full-corpus `k` is different: the far tail of a cosine ranking is dense
    // with near-ties, and the prefix stopping early there is arithmetic, not a
    // defect — which is why the ratio is measured only over small `k`.)
    let (mut small_k_total, mut small_k_full) = (0usize, 0usize);

    for metric in METRICS {
        for dim in DIMS {
            let mut rng = Rng(0xA5A5_0000 ^ dim as u64);
            let n = 200;
            let vectors: Vec<Vec<f32>> = (0..n).map(|_| rng.vector(dim)).collect();
            // Ids deliberately neither contiguous nor zero-based: an off-by-one
            // that indexes the slot rather than reading `ext_ids` would still
            // look right with 0..n.
            let ids: Vec<u64> = (0..n).map(|i| (i as u64) * 13 + 7).collect();

            let index = FlatIndex::new(dim, metric).unwrap();
            let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
            index.add_batch(&ids, &flat).unwrap();
            assert_eq!(index.len(), n);

            for _ in 0..12 {
                let query = rng.vector(dim);
                let ranked = reference_ranking(metric, &ids, &vectors, &query);

                for k in [1usize, 5, 10, n, n + 3] {
                    let hits = index.search(&query, k).unwrap();
                    assert_eq!(
                        hits.len(),
                        k.min(n),
                        "{metric:?} dim={dim} k={k}: wrong result count"
                    );

                    // Distances are the engine's own f32 arithmetic, so they
                    // are checked against the reference by value, not equality.
                    for (rank, hit) in hits.iter().enumerate() {
                        let want = ranked
                            .iter()
                            .find(|(id, _)| *id == hit.id)
                            .unwrap_or_else(|| {
                                panic!("{metric:?} dim={dim}: unknown id {}", hit.id)
                            })
                            .1;
                        let scale = want.abs().max(1.0);
                        assert!(
                            ((hit.distance as f64) - want).abs() / scale < 1e-5,
                            "{metric:?} dim={dim} rank={rank}: distance {} vs reference {want}",
                            hit.distance
                        );
                    }

                    // Nearest first, always.
                    assert!(
                        hits.windows(2).all(|w| w[0].distance <= w[1].distance
                            || w[0].distance.is_nan()
                            || w[1].distance.is_nan()),
                        "{metric:?} dim={dim} k={k}: results are not sorted"
                    );

                    let got: Vec<u64> = hits.iter().map(|h| h.id).collect();
                    let prefix = decisive_prefix(&ranked, k.min(n));
                    assert_eq!(
                        got[..prefix],
                        ranked
                            .iter()
                            .take(prefix)
                            .map(|(id, _)| *id)
                            .collect::<Vec<_>>()[..],
                        "{metric:?} dim={dim} k={k}: the first {prefix} ranks differ \
                         from the f64 reference"
                    );
                    decided += prefix;
                    if k <= 10 {
                        small_k_total += 1;
                        small_k_full += usize::from(prefix == k.min(n));
                    }
                }
            }
        }
    }

    // A run that asserted little would prove little.
    assert!(decided > 100_000, "only {decided} decisively ordered ranks");
    assert!(
        small_k_full * 100 >= small_k_total * 95,
        "only {small_k_full} of {small_k_total} small-k queries had their whole \
         top-k decided by the reference"
    );
}

/// Equal distances come back in ascending id order, on every metric.
///
/// `SearchResult`'s `Ord` promises this and `flat/topk.rs` is built around it,
/// but a heap is free to emit ties in any order — so the promise needs a case
/// where every candidate really is tied.
#[test]
fn exact_ties_are_returned_in_ascending_id_order() {
    for metric in METRICS {
        // The same vector under sixteen ids. Points on a circle are *not* good
        // enough: `cos`/`sin` rounded to f32 put the four axis points at
        // exactly 1.0 and the rest a ulp either side, so the engine ordered
        // them by distance — correctly — and the test caught its own
        // construction rather than a tie. Identical vectors are tied under
        // every metric, exactly, with no rounding to argue about.
        let n = 16usize;
        let vector = [0.25f32, -0.5, 0.75, 1.0];
        // Ids inserted in descending order, so insertion order cannot pass for
        // id order.
        let ids: Vec<u64> = (0..n as u64).map(|i| 900 - i).collect();

        let index = FlatIndex::new(vector.len(), metric).unwrap();
        let flat: Vec<f32> = ids.iter().flat_map(|_| vector).collect();
        index.add_batch(&ids, &flat).unwrap();

        let query = [0.1f32, 0.2, -0.3, 0.4];
        let hits = index.search(&query, n).unwrap();
        assert!(
            hits.windows(2)
                .all(|w| w[0].distance.to_bits() == w[1].distance.to_bits()),
            "{metric:?}: the fixture is not actually tied: {:?}",
            hits.iter().map(|h| h.distance).collect::<Vec<_>>()
        );
        let got: Vec<u64> = hits.iter().map(|h| h.id).collect();
        let mut want = ids.clone();
        want.sort_unstable();
        assert_eq!(got, want, "{metric:?}: tied results are not id-ordered");

        // And the same holds when only part of the tied set is asked for:
        // the k smallest ids, not an arbitrary k of them.
        let top = index.search(&query, 4).unwrap();
        assert_eq!(
            top.iter().map(|h| h.id).collect::<Vec<_>>(),
            want[..4],
            "{metric:?}: truncating a tie did not keep the lowest ids"
        );
    }
}

/// The mapped index answers from the file exactly as the reference does.
///
/// `disk_tests` already compares it to `FlatIndex`; this holds it to the
/// independent ranking instead, so the two exact indexes cannot agree on a
/// wrong answer.
#[cfg(feature = "disk")]
#[test]
fn disk_search_reproduces_the_independent_ranking() {
    for metric in METRICS {
        for dim in [3usize, 8, 17, 64] {
            let mut rng = Rng(0xD15C_0000 ^ dim as u64);
            let n = 150;
            let vectors: Vec<Vec<f32>> = (0..n).map(|_| rng.vector(dim)).collect();
            let ids: Vec<u64> = (0..n).map(|i| (i as u64) * 5 + 3).collect();

            let mut builder = DiskIndexBuilder::new(dim, metric).unwrap();
            for (&id, v) in ids.iter().zip(&vectors) {
                builder.add(id, v).unwrap();
            }
            let path = std::env::temp_dir().join(format!(
                "vanedb-search-correctness-{}-{metric:?}-{dim}.vndb",
                std::process::id()
            ));
            builder.save(&path).unwrap();
            // SAFETY: this test is the only writer, and it does not touch the
            // file again until the mapping is dropped below.
            let index = unsafe { DiskIndex::open(&path) }.unwrap();
            assert_eq!(index.len(), n);
            assert_eq!(index.metric(), metric);

            for _ in 0..10 {
                let query = rng.vector(dim);
                let ranked = reference_ranking(metric, &ids, &vectors, &query);
                let prefix = decisive_prefix(&ranked, 8);
                let got: Vec<u64> = index
                    .search(&query, 8)
                    .unwrap()
                    .iter()
                    .map(|h| h.id)
                    .collect();
                let want: Vec<u64> = ranked.iter().take(prefix).map(|(id, _)| *id).collect();
                assert_eq!(
                    got[..prefix],
                    want[..],
                    "{metric:?} dim={dim}: mapped ranking differs"
                );
            }

            // Stored vectors survive the round trip byte for byte.
            for (&id, v) in ids.iter().zip(&vectors) {
                assert_eq!(index.get(id).unwrap().as_ref(), v.as_slice());
            }
            drop(index);
            std::fs::remove_file(&path).ok();
        }
    }
}

/// The graph finds the true neighbours, not merely plausible ones.
///
/// Recall against `FlatIndex` cannot distinguish "the graph is good" from "both
/// are wrong in the same way". Held against the independent ranking, and with a
/// beam wide enough that approximation is not an excuse, the graph must reach
/// the exact answer.
#[test]
fn the_graph_reaches_the_exact_answer_with_a_wide_beam() {
    for (seed, metric) in METRICS.iter().enumerate() {
        let metric = *metric;
        let dim = 24usize;
        let n = 600usize;
        let mut rng = Rng(0x9F1B_0000 ^ seed as u64);
        let vectors: Vec<Vec<f32>> = (0..n).map(|_| rng.vector(dim)).collect();
        let ids: Vec<u64> = (0..n).map(|i| (i as u64) * 3 + 11).collect();

        let index = ApproxIndex::builder(dim, metric)
            .capacity(n)
            .build()
            .unwrap();
        let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
        index.add_batch(&ids, &flat).unwrap();
        assert_eq!(index.len(), n);

        let wide = SearchParams::new().ef_search(n);
        let mut compared = 0usize;
        for _ in 0..25 {
            let query = rng.vector(dim);
            let ranked = reference_ranking(metric, &ids, &vectors, &query);
            let prefix = decisive_prefix(&ranked, 10);
            if prefix == 0 {
                continue;
            }
            let got: Vec<u64> = index
                .search_with(&query, 10, &wide)
                .unwrap()
                .iter()
                .map(|h| h.id)
                .collect();
            let want: Vec<u64> = ranked.iter().take(prefix).map(|(id, _)| *id).collect();
            assert_eq!(
                got[..prefix],
                want[..],
                "{metric:?}: a beam as wide as the corpus still missed the true nearest"
            );
            compared += 1;
        }
        assert!(
            compared >= 20,
            "{metric:?}: only {compared} decisive queries"
        );
    }
}

/// Clusters far enough apart that an approximate index has no excuse.
///
/// This is the shape real corpora have — embeddings group by topic — and it is
/// the case a user would notice going wrong. At the default `ef_search`, every
/// neighbour of a cluster centre must come from that cluster.
#[test]
fn separable_clusters_are_searched_correctly_at_the_default_beam() {
    let dim = 16usize;
    let clusters = 8usize;
    let per = 60usize;
    let mut rng = Rng(0x0C1D_5747);

    let centres: Vec<Vec<f32>> = (0..clusters)
        .map(|_| rng.vector(dim).iter().map(|v| v * 50.0).collect())
        .collect();
    let mut ids = Vec::new();
    let mut vectors = Vec::new();
    for (c, centre) in centres.iter().enumerate() {
        for j in 0..per {
            let jitter = rng.vector(dim);
            vectors.push(
                centre
                    .iter()
                    .zip(&jitter)
                    .map(|(m, d)| m + d * 0.01)
                    .collect::<Vec<f32>>(),
            );
            ids.push((c * per + j) as u64);
        }
    }

    let index = ApproxIndex::builder(dim, Metric::L2)
        .capacity(ids.len())
        .build()
        .unwrap();
    let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
    index.add_batch(&ids, &flat).unwrap();

    for (c, centre) in centres.iter().enumerate() {
        let hits = index.search(centre, per).unwrap();
        assert_eq!(hits.len(), per);
        for hit in &hits {
            assert_eq!(
                hit.id as usize / per,
                c,
                "cluster {c}: id {} came from cluster {}",
                hit.id,
                hit.id as usize / per
            );
        }
    }
}
