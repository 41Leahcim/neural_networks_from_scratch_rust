#![warn(clippy::pedantic, clippy::nursery)]
#![allow(clippy::cast_precision_loss)]

use rayon::prelude::{IntoParallelIterator, ParallelIterator};

/// Generates a vector containing values with linear space
///
/// # Arguments
/// ```start```: the start value (inclusive)
/// ```stop```: the end value (exclusive)
/// ```samples```: the number of values
///
/// # Returns
/// A vector containing linear spaced values
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn linspace(start: f64, stop: f64, samples: usize) -> Vec<f64> {
    // Calculate the step size
    let step_size = (stop - start) / samples as f64;

    // Return the result
    (0..samples)
        .map(|i| (i as f64).mul_add(step_size, start))
        .collect()
}

/// Creates a matrix filled with random numbers
///
/// # Arguments
/// ```rows```: the outer-size of the matrix
/// ```columns```: the inner-size of the matrix
///
/// # Returns
/// A matrix containing random numbers
#[must_use]
pub fn randn_matrix(rows: usize, columns: usize) -> Vec<Vec<f64>> {
    // Create a matrix filled with random values between -1 and 1
    (0..rows)
        .into_par_iter()
        .map(|_| {
            (0..columns)
                .map(|_| rand::random::<f64>().mul_add(2.0, -1.0))
                .collect()
        })
        .collect()
}
