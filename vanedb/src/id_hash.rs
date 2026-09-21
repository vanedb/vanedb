//! Hasher for the internal `u64` id maps (RFC 0010, #109).
//!
//! Every index keeps a `u64 -> slot` map. Its keys are caller-chosen ids,
//! and the default `RandomState` runs SipHash-1-3 over them: a keyed,
//! HashDoS-resistant function that costs tens of nanoseconds per probe on
//! an eight-byte key. That was the whole gap on the `store_add` benchmark
//! row, where the C++ engine hashes the same ids by identity.
//!
//! Raw identity is not an option here. hashbrown takes the bucket index
//! from the low bits of the hash and the 7-bit control tag from the top
//! bits, so a caller whose ids are multiples of `2^20` -- or any other
//! family sharing low bits -- would land every key in one bucket and turn
//! each insert into a linear probe. The finaliser below is a Fibonacci
//! multiply (spreads every input bit upward), a fold of the well-mixed
//! high half into the low half, and a second multiply-fold so the tag bits
//! depend on the folded low half too. Two multiplies and two shifts: a few
//! cycles, no key material, and every bit of the output depends on every
//! bit of the input. `adversarial_families_insert_within_bound` below
//! pins that property with the key families that defeat plain identity.
//!
//! This is not a security boundary. An adversary who can choose ids can
//! still craft collisions for any unkeyed function; the maps are bounded
//! by the caller's own inserts and the engine never hashes untrusted keys
//! it did not store, so that is the same exposure as the vector storage
//! itself.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};

/// Map from external id to whatever the index keeps per id.
pub(crate) type IdMap<V> = HashMap<u64, V, IdBuildHasher>;

/// Set of external ids.
pub(crate) type IdSet = HashSet<u64, IdBuildHasher>;

/// Mixes one `u64` key. Bijective, so distinct keys never collide in the
/// full 64-bit hash; the bucket and tag bits are drawn from it, which is
/// where the mixing matters.
#[inline]
pub(crate) fn mix(key: u64) -> u64 {
    // 2^64 / phi: the Fibonacci hashing constant. Odd, so the multiply is
    // a bijection; its top bits are a good hash of the whole key, its low
    // bits are still only a function of the key's low bits.
    let h = key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    // Fold the top half down so the bucket index sees the mixed bits.
    let h = h ^ (h >> 32);
    // A second odd multiply and fold so the top 7 bits (hashbrown's tag)
    // depend on the folded low half as well. Constant from the `moremur`
    // finaliser family; any odd constant with a balanced bit pattern does.
    let h = h.wrapping_mul(0x3C79_AC49_2BA7_B653);
    h ^ (h >> 32)
}

/// `BuildHasher` for [`IdMap`] and [`IdSet`]. Stateless, so `Default` and
/// `Clone` are free and serde can deserialise straight into the map.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct IdBuildHasher;

impl BuildHasher for IdBuildHasher {
    type Hasher = IdHasher;

    #[inline]
    fn build_hasher(&self) -> IdHasher {
        IdHasher(0)
    }
}

/// The `Hasher` behind [`IdBuildHasher`]. `u64` hashes through
/// `write_u64`, which is the only path the id maps take; the generic
/// `write` exists because the trait requires it and folds bytes through the
/// same mixer so a differently-typed key would still be well distributed.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct IdHasher(u64);

impl Hasher for IdHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.0 = mix(self.0 ^ u64::from_le_bytes(word));
        }
    }

    #[inline]
    fn write_u64(&mut self, key: u64) {
        self.0 = mix(self.0 ^ key);
    }

    #[inline]
    fn write_usize(&mut self, key: usize) {
        self.write_u64(key as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const N: u64 = 1_000_000;

    /// SplitMix64, written out so the random family is the same on every
    /// platform and every dependency version.
    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// The random baseline plus the three structured families that defeat
    /// plain identity hashing: sequential ids, ids sharing their low 20
    /// bits, and ids living only in the top 20 bits.
    ///
    /// The random keys are returned in draw order. Collecting them out of
    /// an `IdSet` with this same hasher would hand them back in bucket order,
    /// and inserting keys in bucket order is cache-sequential, which makes
    /// the baseline artificially fast and eats into the margin the timing
    /// test grants the structured families.
    fn families() -> (Vec<u64>, [(&'static str, Vec<u64>); 3]) {
        let mut state = 0x5EED_0010;
        let mut seen = IdSet::with_capacity_and_hasher(N as usize, IdBuildHasher);
        let mut random = Vec::with_capacity(N as usize);
        while random.len() < N as usize {
            let key = splitmix(&mut state);
            if seen.insert(key) {
                random.push(key);
            }
        }
        let structured = [
            ("sequential", (0..N).collect()),
            (
                "low bits shared (multiples of 2^20)",
                (0..N).map(|i| i << 20).collect(),
            ),
            (
                "high bits only (multiples of 2^44)",
                (0..N).map(|i| i << 44).collect(),
            ),
        ];
        (random, structured)
    }

    /// Inserts every key, glancing at the clock every 4096 inserts and
    /// giving up once `ceiling` has passed. `None` means the ceiling was hit:
    /// a hasher that has collapsed to a linear probe is quadratic at 1M keys
    /// and would otherwise run for minutes, turning a test failure into a CI
    /// timeout.
    fn insert_all_within(keys: &[u64], ceiling: Duration) -> Option<Duration> {
        const CHECK_EVERY: usize = 1 << 12;
        let start = Instant::now();
        let mut map = IdMap::default();
        for (slot, &key) in keys.iter().enumerate() {
            assert!(map.insert(key, slot).is_none(), "keys must be distinct");
            if slot % CHECK_EVERY == CHECK_EVERY - 1 && start.elapsed() > ceiling {
                return None;
            }
        }
        let elapsed = start.elapsed();
        assert_eq!(map.len(), keys.len());
        Some(elapsed)
    }

    /// Wall time of the fastest of three full inserts, each capped at
    /// `ceiling`. The minimum, not the mean: a scheduler hiccup only ever
    /// adds time, so the minimum is the run least contaminated by the host.
    /// `None` as soon as any run hits the ceiling.
    fn time_inserts(keys: &[u64], ceiling: Duration) -> Option<Duration> {
        let mut best: Option<Duration> = None;
        for _ in 0..3 {
            let run = insert_all_within(keys, ceiling)?;
            best = Some(best.map_or(run, |b| b.min(run)));
        }
        best
    }

    /// A `u64` key reaches the map through `Hash::hash`, which calls
    /// `write_u64` exactly once on a fresh hasher, so the map's hash of a key
    /// is `mix(key)` and nothing else. A `usize` key takes `write_usize` and
    /// must agree with the same value as a `u64`.
    #[test]
    fn u64_and_usize_keys_hash_to_the_mixer_output() {
        let build = IdBuildHasher;
        for key in [0u64, 1, 7, u64::MAX, 0x9E37_79B9_7F4A_7C15, 1 << 63] {
            assert_eq!(build.hash_one(key), mix(key), "u64 {key:#x}");
            if let Ok(narrow) = usize::try_from(key) {
                assert_eq!(build.hash_one(narrow), mix(key), "usize {key:#x}");
            }
        }
        let mut hasher = build.build_hasher();
        hasher.write_usize(42);
        assert_eq!(hasher.finish(), mix(42));
    }

    /// The generic byte path folds each little-endian 8-byte word through
    /// the mixer, zero-padding the tail. It is never taken by the id maps,
    /// but the trait requires it, and a differently-typed key must still be
    /// distributed rather than truncated or ignored.
    #[test]
    fn generic_write_folds_every_word_and_the_padded_tail() {
        fn fold(bytes: &[u8]) -> u64 {
            let mut hasher = IdBuildHasher.build_hasher();
            hasher.write(bytes);
            hasher.finish()
        }
        assert_eq!(fold(&[]), 0, "no bytes leave the fresh state alone");
        // One full word is the same as `write_u64` of that word.
        let word = 0x0102_0304_0506_0708u64;
        assert_eq!(fold(&word.to_le_bytes()), mix(word));
        // A short tail is zero-padded, so it equals the padded word...
        assert_eq!(fold(&[0x08, 0x07, 0x06]), mix(0x0006_0708));
        // ...and every byte position in the tail is significant.
        assert_ne!(fold(&[0x08, 0x07, 0x06]), fold(&[0x08, 0x07, 0x05]));
        // Two words chain: the second is mixed against the first's output.
        let mut two = word.to_le_bytes().to_vec();
        two.extend_from_slice(&0xFFu64.to_le_bytes());
        assert_eq!(fold(&two), mix(mix(word) ^ 0xFF));
        // A trailing zero byte still changes the length, and so the hash.
        let mut padded = word.to_le_bytes().to_vec();
        padded.push(0);
        assert_ne!(fold(&padded), fold(&word.to_le_bytes()));
        // Str keys go through `write` plus the `0xff` terminator and must
        // be distinct for distinct strings.
        assert_ne!(
            IdBuildHasher.hash_one("alpha"),
            IdBuildHasher.hash_one("alphb")
        );
    }

    /// The builder is stateless: every copy, clone and default builds a
    /// hasher that starts from zero and agrees with every other. The
    /// `clone` and `default` calls on `Copy` unit types are the point: they
    /// run the derived impls, which serde and `with_capacity_and_hasher`
    /// rely on and nothing else in the crate calls directly.
    #[test]
    #[allow(clippy::clone_on_copy, clippy::default_constructed_unit_structs)]
    fn build_hasher_is_stateless_and_freely_copied() {
        let a = IdBuildHasher;
        let b = a;
        let c = a.clone();
        let d = IdBuildHasher::default();
        assert_eq!(IdHasher::default().finish(), 0);
        assert_eq!(a.build_hasher().finish(), 0);
        let key = 0xDEAD_BEEFu64;
        assert_eq!(a.hash_one(key), b.hash_one(key));
        assert_eq!(b.hash_one(key), c.hash_one(key));
        assert_eq!(c.hash_one(key), d.hash_one(key));
        let mut hasher = a.build_hasher();
        hasher.write_u64(key);
        let copy = hasher;
        let clone = hasher.clone();
        assert_eq!(copy.finish(), mix(key));
        assert_eq!(clone.finish(), mix(key));
        assert_eq!(format!("{a:?}"), "IdBuildHasher");
        assert_eq!(format!("{:?}", IdHasher::default()), "IdHasher(0)");
        // Two maps built from separate defaults see the same buckets: an
        // `IdMap` moved between them keeps every lookup.
        let mut map: IdMap<u8> = IdMap::with_capacity_and_hasher(4, d);
        map.insert(key, 1);
        assert_eq!(map.get(&key), Some(&1));
        assert_eq!(map.hasher().hash_one(key), mix(key));
    }

    #[test]
    fn mix_is_a_bijection_on_small_inputs_and_moves_every_bit() {
        // Distinct on a dense range: the multiply is odd and the folds are
        // invertible, so no two keys share a full hash.
        let mut seen = std::collections::HashSet::new();
        for key in 0..100_000u64 {
            assert!(seen.insert(mix(key)));
        }
        // Flipping any single input bit must change both the bucket bits
        // (low) and the tag bits (top 7) of at least one probe key: the
        // property plain identity and a lone multiply each lack.
        let probe = 0x0123_4567_89AB_CDEFu64;
        let base = mix(probe);
        for bit in 0..64 {
            let flipped = mix(probe ^ (1u64 << bit));
            assert_ne!(
                base & 0xFFFF,
                flipped & 0xFFFF,
                "bit {bit} left the low half alone"
            );
        }
        let tags: std::collections::HashSet<u64> = (0..64)
            .map(|bit| mix(probe ^ (1u64 << bit)) >> 57)
            .collect();
        assert!(
            tags.len() > 16,
            "tag bits barely move: {} distinct",
            tags.len()
        );
    }

    /// Structured families must insert within this factor of random keys.
    const BOUND: f64 = 3.0;
    /// A single run is abandoned past this multiple of the random baseline...
    const CEILING_FACTOR: u32 = 20;
    /// ...but never sooner than this, so a fast baseline cannot starve a run.
    const CEILING_FLOOR: Duration = Duration::from_secs(2);

    /// Judges one family: `None` is a run that passed the ceiling, otherwise
    /// the ratio to the random baseline must be within [`BOUND`]. Returns
    /// the ratio. Factored out of the timing test so both failure arms can
    /// be exercised with numbers instead of a collapsed hasher.
    fn check_family_time(
        name: &str,
        elapsed: Option<Duration>,
        baseline: Duration,
        ceiling: Duration,
    ) -> f64 {
        let Some(elapsed) = elapsed else {
            panic!(
                "{name}: a run passed the {:.3}s ceiling ({CEILING_FACTOR}x the random \
                 baseline of {:.3}s); the id hasher has collapsed on structured keys",
                ceiling.as_secs_f64(),
                baseline.as_secs_f64()
            );
        };
        let ratio = elapsed.as_secs_f64() / baseline.as_secs_f64();
        println!(
            "{name}: {:.3}s, {ratio:.2}x the baseline",
            elapsed.as_secs_f64()
        );
        assert!(
            ratio <= BOUND,
            "{name}: {:.3}s is {ratio:.2}x the random baseline of {:.3}s; \
             the id hasher is degenerating on structured keys",
            elapsed.as_secs_f64(),
            baseline.as_secs_f64()
        );
        ratio
    }

    /// RFC 0010: 1M sequential ids and 1M ids sharing their low bits must
    /// insert within a bounded factor of 1M random ids.
    ///
    /// The comparison is a ratio measured in one process against the same
    /// map type, so it is independent of the build profile and of the host's
    /// absolute speed. Under plain identity hashing the low-bits family
    /// takes minutes rather than a fraction of a second, so the bound only
    /// has to separate "hashed" from "degenerate"; a 3x margin leaves room
    /// for a loaded CI runner while catching a collapse by two orders of
    /// magnitude. Each family is judged as soon as it finishes, and a run
    /// that passes the ceiling is abandoned there, so a collapsed hasher
    /// fails in seconds rather than hanging the job.
    /// `every_family_spreads_over_buckets_and_tags_like_random_keys` is the
    /// timing-free statement of the same property.
    #[test]
    fn adversarial_families_insert_within_bound() {
        let (random, structured) = families();

        // Warm the allocator and the code once so the first timed family is
        // not paying for page faults the others do not.
        let _ = insert_all_within(&random[..N as usize / 8], Duration::MAX);

        let baseline = time_inserts(&random, Duration::MAX).unwrap();
        let ceiling = (baseline * CEILING_FACTOR).max(CEILING_FLOOR);
        println!(
            "random baseline {:.3}s, per-run ceiling {:.3}s",
            baseline.as_secs_f64(),
            ceiling.as_secs_f64()
        );
        for (name, keys) in &structured {
            check_family_time(name, time_inserts(keys, ceiling), baseline, ceiling);
        }
    }

    /// The abandon path of the timed insert: a ceiling that has already
    /// passed at the first clock check returns `None` from the run and from
    /// the min-of-three around it, and the same keys complete under an
    /// unreachable ceiling.
    #[test]
    fn a_run_past_the_ceiling_is_abandoned() {
        let keys: Vec<u64> = (0..2 * 4096).collect();
        assert_eq!(insert_all_within(&keys, Duration::ZERO), None);
        assert_eq!(time_inserts(&keys, Duration::ZERO), None);
        assert!(insert_all_within(&keys, Duration::MAX).is_some());
        assert!(time_inserts(&keys, Duration::MAX).is_some());
    }

    #[test]
    fn family_check_accepts_a_ratio_within_the_bound() {
        let ratio = check_family_time(
            "healthy",
            Some(Duration::from_secs(2)),
            Duration::from_secs(1),
            Duration::from_secs(20),
        );
        assert_eq!(ratio, 2.0);
    }

    #[test]
    #[should_panic(expected = "has collapsed on structured keys")]
    fn family_check_rejects_an_abandoned_run() {
        check_family_time(
            "abandoned",
            None,
            Duration::from_secs(1),
            Duration::from_secs(20),
        );
    }

    #[test]
    #[should_panic(expected = "is degenerating on structured keys")]
    fn family_check_rejects_a_ratio_past_the_bound() {
        check_family_time(
            "slow",
            Some(Duration::from_secs(4)),
            Duration::from_secs(1),
            Duration::from_secs(20),
        );
    }

    /// Low hash bits hashbrown turns into a bucket index, for the
    /// occupancy check.
    const BUCKET_BITS: u32 = 21;
    const BUCKETS: usize = 1 << BUCKET_BITS;
    /// hashbrown's control tag is the top 7 bits.
    const TAGS: usize = 1 << 7;
    const BUCKET_TOLERANCE: f64 = 0.05;
    const TAG_TOLERANCE: f64 = 0.05;

    /// What one family leaves in the bucket and tag space.
    struct Spread {
        distinct: usize,
        tags: [u64; TAGS],
    }

    fn spread(keys: &[u64]) -> Spread {
        let mut occupied = vec![0u64; BUCKETS / 64];
        let mut tags = [0u64; TAGS];
        for &key in keys {
            let hash = mix(key);
            let bucket = hash as usize & (BUCKETS - 1);
            occupied[bucket / 64] |= 1 << (bucket % 64);
            tags[(hash >> (64 - 7)) as usize] += 1;
        }
        let distinct = occupied.iter().map(|w| w.count_ones() as usize).sum();
        Spread { distinct, tags }
    }

    /// The uniform expectation for `keys` keys: distinct buckets within
    /// [`BUCKET_TOLERANCE`] of `BUCKETS * (1 - e^(-keys / BUCKETS))`, and
    /// every tag within [`TAG_TOLERANCE`] of `keys / TAGS`. Factored out so
    /// each failure arm can be exercised with numbers.
    fn check_spread(name: &str, spread: &Spread, keys: usize) {
        let expected_distinct = BUCKETS as f64 * (1.0 - (-(keys as f64) / BUCKETS as f64).exp());
        let expected_per_tag = keys as f64 / TAGS as f64;
        let distinct = spread.distinct;
        let (min_tag, max_tag) = spread
            .tags
            .iter()
            .fold((u64::MAX, 0), |(lo, hi), &c| (lo.min(c), hi.max(c)));
        println!(
            "{name}: {distinct} of {BUCKETS} buckets (expected {expected_distinct:.0}), \
             tag counts {min_tag}..={max_tag} (expected {expected_per_tag:.0})"
        );
        let bucket_error = (distinct as f64 - expected_distinct).abs() / expected_distinct;
        assert!(
            bucket_error <= BUCKET_TOLERANCE,
            "{name}: {distinct} distinct buckets of {BUCKETS}, expected about \
             {expected_distinct:.0}; the low hash bits are not uniform"
        );
        for (tag, &count) in spread.tags.iter().enumerate() {
            let tag_error = (count as f64 - expected_per_tag).abs() / expected_per_tag;
            assert!(
                tag_error <= TAG_TOLERANCE,
                "{name}: tag {tag} holds {count} keys, expected about \
                 {expected_per_tag:.0}; the top 7 hash bits are not balanced"
            );
        }
    }

    /// The timing-free complement of `adversarial_families_insert_within_bound`:
    /// what hashbrown actually consumes from the hash is the low bits for the
    /// bucket and the top 7 bits for the control tag, so every family must
    /// spread over both the way random keys do.
    ///
    /// 1M keys into 2^21 buckets leave `2^21 * (1 - e^(-1M/2^21))`, about
    /// 795k, distinct buckets when the hash is uniform; a family that maps
    /// to a few hundred buckets (a single-fold mixer sends the `i << 44`
    /// family to 512 of them) misses by orders of magnitude. The 128 tags
    /// must each hold their share within a few percent, or a full bucket
    /// group would need a probe per key rather than one per group.
    #[test]
    fn every_family_spreads_over_buckets_and_tags_like_random_keys() {
        let (random, structured) = families();
        let all = std::iter::once(("random", &random))
            .chain(structured.iter().map(|(name, keys)| (*name, keys)));
        for (name, keys) in all {
            check_spread(name, &spread(keys), keys.len());
        }
    }

    /// A spread that is uniform on paper: exactly the expected bucket count
    /// and every tag at its share.
    fn ideal_spread(keys: usize) -> Spread {
        let expected_distinct = BUCKETS as f64 * (1.0 - (-(keys as f64) / BUCKETS as f64).exp());
        Spread {
            distinct: expected_distinct.round() as usize,
            tags: [(keys / TAGS) as u64; TAGS],
        }
    }

    #[test]
    fn spread_check_accepts_the_uniform_expectation() {
        check_spread("ideal", &ideal_spread(N as usize), N as usize);
    }

    /// The single-fold mutant's signature: the `i << 44` family in 512
    /// buckets.
    #[test]
    #[should_panic(expected = "the low hash bits are not uniform")]
    fn spread_check_rejects_sparse_buckets() {
        let mut sparse = ideal_spread(N as usize);
        sparse.distinct = 512;
        check_spread("sparse", &sparse, N as usize);
    }

    #[test]
    #[should_panic(expected = "the top 7 hash bits are not balanced")]
    fn spread_check_rejects_an_unbalanced_tag() {
        let mut lopsided = ideal_spread(N as usize);
        lopsided.tags[77] = 0;
        check_spread("lopsided", &lopsided, N as usize);
    }
}
