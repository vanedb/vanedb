use std::cmp::Ordering;

use crate::error::{Result, VaneError};

#[inline]
pub(crate) fn validate_finite(values: &[f32], input: &'static str) -> Result<()> {
    if all_finite(values) {
        Ok(())
    } else {
        Err(VaneError::NonFiniteValue { input })
    }
}

/// Floats per branch-free block in [`all_finite`]. Large enough that the
/// inner loop amortises the per-block check and vectorises; small enough
/// that a batch whose first component is non-finite stops after one block
/// rather than scanning the whole slice.
const FINITE_BLOCK: usize = 256;

/// `values.iter().all(f32::is_finite)`, written so the compiler vectorises it
/// (RFC 0010, #77).
///
/// `is_finite` is `(bits & 0x7fff_ffff) < 0x7f80_0000`, or, per lane, "not
/// every exponent bit set". The short-circuiting `all` decides that one
/// element at a time, and a data-dependent early exit is exactly what stops
/// the loop from being widened. Here each block folds the per-lane test into
/// one accumulator with a bitwise OR -- no exit inside the block, so the
/// compiler emits an and/compare/or over as many lanes as the target has --
/// and the exit is checked once per block. The result is identical to the
/// element-wise form for every input, including NaN payloads and both
/// infinities, and an empty slice is finite.
#[inline]
pub(crate) fn all_finite(values: &[f32]) -> bool {
    const EXPONENT: u32 = 0x7f80_0000;
    values.chunks(FINITE_BLOCK).all(|block| {
        let non_finite = block.iter().fold(0u32, |acc, value| {
            acc | ((value.to_bits() & EXPONENT) == EXPONENT) as u32
        });
        non_finite == 0
    })
}

/// The query preamble every search shares: dimension, then finiteness, then
/// `k`.
///
/// The *order* matters where a caller can see both failures at once: a
/// non-finite query of the wrong length must report the mismatch, not the
/// non-finite value. That is only the Python surfaces — both C ABIs take a
/// bare pointer with no length, so neither can raise the dimension error.
/// Nothing pins it either: `non_finite_vectors.tsv` varies the value and not
/// the length, so the engines agree by construction rather than by contract.
///
/// The check itself was stated independently in three search paths, and a
/// fourth index type would have made a fourth copy.
#[inline]
pub(crate) fn validate_query(query: &[f32], dim: usize, k: usize) -> Result<()> {
    validate_dimension(query, dim)?;
    validate_finite(query, "query")?;
    if k == 0 {
        return Err(VaneError::InvalidK);
    }
    Ok(())
}

/// The add preamble: dimension, then finiteness. Four copies before this.
#[inline]
pub(crate) fn validate_vector(vector: &[f32], dim: usize) -> Result<()> {
    validate_dimension(vector, dim)?;
    validate_finite(vector, "vector")
}

#[inline]
fn validate_dimension(values: &[f32], dim: usize) -> Result<()> {
    if values.len() != dim {
        return Err(VaneError::DimensionMismatch {
            expected: dim,
            got: values.len(),
        });
    }
    Ok(())
}

/// Total distance order used by every top-k path: finite values first,
/// then infinities, then NaNs ordered by `f32::total_cmp`.
#[inline]
pub(crate) fn compare_distances(left: f32, right: f32) -> Ordering {
    match left.partial_cmp(&right) {
        Some(Ordering::Less) if left == f32::NEG_INFINITY => Ordering::Greater,
        Some(Ordering::Greater) if right == f32::NEG_INFINITY => Ordering::Less,
        Some(order) => order,
        None => match (left.is_nan(), right.is_nan()) {
            (false, true) => Ordering::Less,
            (true, false) => Ordering::Greater,
            _ => left.total_cmp(&right),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The block form must agree with the element-wise form on every
    /// position a non-finite value can take relative to the block size,
    /// and on every kind of non-finite value.
    #[test]
    fn all_finite_matches_the_element_wise_form_at_every_block_offset() {
        let poisons = [
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7f80_0001),
            f32::from_bits(0xffff_ffff),
        ];
        let benign = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            1e-45,
        ];
        assert!(all_finite(&[]));
        for len in [1, 7, 8, 255, 256, 257, 512, 1000] {
            let mut values: Vec<f32> = (0..len).map(|i| benign[i % benign.len()]).collect();
            assert!(all_finite(&values), "len {len}: all-benign slice");
            for at in [0, len / 2, len - 1] {
                for poison in poisons {
                    let keep = values[at];
                    values[at] = poison;
                    assert_eq!(
                        all_finite(&values),
                        values.iter().all(|v| v.is_finite()),
                        "len {len}, poison {poison:?} at {at}"
                    );
                    assert!(!all_finite(&values));
                    values[at] = keep;
                }
            }
        }
    }

    #[test]
    fn distances_preserve_finite_first_order_and_signed_zero_equivalence() {
        let ordered = [
            -f32::MAX,
            -1.0,
            0.0,
            1.0,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0xffff_ffff),
            f32::from_bits(0xffc0_0000),
            f32::from_bits(0x7fc0_0000),
            f32::from_bits(0x7fff_ffff),
        ];
        for (i, &left) in ordered.iter().enumerate() {
            for (j, &right) in ordered.iter().enumerate() {
                assert_eq!(compare_distances(left, right), i.cmp(&j));
            }
        }
        assert_eq!(compare_distances(-0.0, 0.0), Ordering::Equal);
        assert_eq!(compare_distances(0.0, -0.0), Ordering::Equal);
    }
}
