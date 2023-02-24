use std::ops::Add;

use rayon::prelude::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator};

#[must_use]
pub fn vectors(left: &[f64], right: &[f64]) -> Vec<f64> {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| left.add(right))
        .collect()
}

#[must_use]
pub fn matrix_vector(left: &[Vec<f64>], right: &[f64], add_to_row: bool) -> Vec<Vec<f64>> {
    if add_to_row {
        assert_eq!(left.len(), right.len());
        left.par_iter()
            .zip(right)
            .map(|(left, right)| left.iter().map(|left| left + right).collect())
            .collect()
    } else {
        left.par_iter()
            .map(|left| {
                assert_eq!(left.len(), right.len());
                left.iter()
                    .zip(right)
                    .map(|(left, right)| left + right)
                    .collect()
            })
            .collect()
    }
}
