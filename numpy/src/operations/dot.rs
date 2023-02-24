use std::ops::Mul;

use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

#[must_use]
pub fn vectors(left: &[f64], right: &[f64]) -> f64 {
    (0..left.len()).map(|i| left[i] * right[i]).sum()
}

#[must_use]
pub fn matrix_vector(left: &[Vec<f64>], right: &[f64]) -> Vec<f64> {
    left.iter().map(|left| vectors(left, right)).collect()
}

#[must_use]
pub fn matrix(left: &[Vec<f64>], right: &[Vec<f64>]) -> Vec<Vec<f64>> {
    left.par_iter()
        .map(|left_vector| {
            assert_eq!(left_vector.len(), left[0].len());
            (0..right[0].len())
                .map(|i| {
                    left_vector
                        .iter()
                        .zip(right)
                        .map(|(left_value, right_vector)| {
                            assert_eq!(left_vector.len(), right.len());
                            assert_eq!(right_vector.len(), right[0].len());
                            left_value.mul(right_vector[i])
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}
