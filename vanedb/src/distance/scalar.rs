use crate::distance::COSINE_EPSILON;

pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

pub fn cosine_distance(_a: &[f32], _b: &[f32]) -> f32 {
    let _ = COSINE_EPSILON; // used in Task 4
    todo!()
}

pub fn dot_distance(_a: &[f32], _b: &[f32]) -> f32 {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        assert_eq!(l2_squared(&a, &a), 0.0);
    }

    #[test]
    fn l2_known_result() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_eq!(l2_squared(&a, &b), 2.0);
    }

    #[test]
    fn l2_single_dimension() {
        let a = vec![3.0];
        let b = vec![7.0];
        assert_eq!(l2_squared(&a, &b), 16.0);
    }
}
