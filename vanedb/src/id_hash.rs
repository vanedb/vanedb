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

    fn insert_all(keys: &[u64]) -> IdMap<usize> {
        let mut map = IdMap::default();
        for (slot, &key) in keys.iter().enumerate() {
            assert!(map.insert(key, slot).is_none(), "keys must be distinct");
        }
        map
    }

    /// Wall time of the fastest of three full inserts. The minimum, not
    /// the mean: a scheduler hiccup only ever adds time, so the minimum is
    /// the run least contaminated by the host.
    fn time_inserts(keys: &[u64]) -> Duration {
        (0..3)
            .map(|_| {
                let start = Instant::now();
                let map = insert_all(keys);
                let elapsed = start.elapsed();
                assert_eq!(map.len(), keys.len());
                elapsed
            })
            .min()
            .unwrap()
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

    /// RFC 0010: 1M sequential ids and 1M ids sharing their low bits must
    /// insert within a bounded factor of 1M random ids.
    ///
    /// The comparison is a ratio measured in one process against the same
    /// map type, so it is independent of the build profile and of the host's
    /// absolute speed. Under plain identity hashing the low-bits family
    /// takes minutes rather than a fraction of a second, so the bound only
    /// has to separate "hashed" from "degenerate"; a 3x margin leaves room
    /// for a loaded CI runner while catching a collapse by two orders of
    /// magnitude.
    #[test]
    fn adversarial_families_insert_within_bound() {
        const BOUND: f64 = 3.0;

        let mut state = 0x5EED_0010;
        let random: Vec<u64> = {
            let mut set = IdSet::default();
            while set.len() < N as usize {
                set.insert(splitmix(&mut state));
            }
            set.into_iter().collect()
        };
        let sequential: Vec<u64> = (0..N).collect();
        let low_bits_shared: Vec<u64> = (0..N).map(|i| i << 20).collect();
        let high_bits_only: Vec<u64> = (0..N).map(|i| i << 44).collect();

        // Warm the allocator and the code once so the first timed family is
        // not paying for page faults the others do not.
        drop(insert_all(&random[..N as usize / 8]));

        let baseline = time_inserts(&random).as_secs_f64();
        for (name, keys) in [
            ("sequential", &sequential),
            ("low bits shared (multiples of 2^20)", &low_bits_shared),
            ("high bits only (multiples of 2^44)", &high_bits_only),
        ] {
            let elapsed = time_inserts(keys).as_secs_f64();
            let ratio = elapsed / baseline;
            assert!(
                ratio <= BOUND,
                "{name}: {elapsed:.3}s is {ratio:.2}x the random baseline of {baseline:.3}s; \
                 the id hasher is degenerating on structured keys"
            );
        }
    }
}
