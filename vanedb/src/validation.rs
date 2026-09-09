use std::cmp::Ordering;

use crate::error::{Result, VaneError};

#[inline]
pub(crate) fn validate_finite(values: &[f32], input: &'static str) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(VaneError::NonFiniteValue { input })
    }
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
