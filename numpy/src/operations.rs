pub mod add;
pub mod dot;

use rayon::prelude::{ParallelIterator, IntoParallelIterator};

// Transpose
#[must_use]
pub fn t(original: &[Vec<f64>]) -> Vec<Vec<f64>> {
    (0..original[0].len()).into_par_iter()
        .map(|i| original.iter().map(|vector| vector[i]).collect())
        .collect()
}
