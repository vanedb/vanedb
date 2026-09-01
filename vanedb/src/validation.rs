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

/// Total distance order used by every top-k path: finite values first,
/// followed by infinities and NaNs in `f32::total_cmp` order.
#[inline]
pub(crate) fn compare_distances(left: f32, right: f32) -> Ordering {
    if left < right {
        return if left == f32::NEG_INFINITY {
            Ordering::Greater
        } else {
            Ordering::Less
        };
    }
    if right < left {
        return if right == f32::NEG_INFINITY {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    match (left.is_nan(), right.is_nan()) {
        (false, false) => Ordering::Equal,
        (false, true) => Ordering::Less,
        (true, false) => Ordering::Greater,
        (true, true) => left.total_cmp(&right),
    }
}
