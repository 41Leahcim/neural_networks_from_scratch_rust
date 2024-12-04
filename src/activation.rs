//! A module containing all activation functions.

use std::cmp::Ordering;

/// A simple activation function.
/// Sets every negative value to 0 without changing other values.
pub struct ReLU;

impl ReLU {
    /// Forwards the input data through the `ReLU` layer, setting every negative value to 0.
    /// Every other value will remain the same.
    pub fn forward<Sample: IntoIterator<Item = f64>, Input: IntoIterator<Item = Sample>>(
        &self,
        inputs: Input,
    ) -> Vec<Vec<f64>> {
        inputs
            .into_iter()
            .map(|sample| sample.into_iter().map(|value| value.max(0.0)).collect())
            .collect()
    }
}

/// A more complex activation function.
/// Puts every value in a curve between 0 and 1.
/// 0 representing to the minimum value, 1 representing the maximum value.
pub struct Softmax;

impl Softmax {
    /// Forwards the input data through the softmax layer.
    /// This puts every value in a curve between 0 and 1.
    /// 0 representing the minimum value, 1 representing the maximum value.
    pub fn forward<Sample: IntoIterator<Item = f64>, Input: IntoIterator<Item = Sample>>(
        &self,
        inputs: Input,
    ) -> Vec<Vec<f64>> {
        inputs
            .into_iter()
            .map(|sample| {
                let sample = sample.into_iter().collect::<Vec<_>>();

                // Get unnormalized probabilities
                let sample_max = sample
                    .iter()
                    .copied()
                    .max_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal))
                    .unwrap_or(1.0);
                let exp_values = sample
                    .into_iter()
                    .map(|value| (value - sample_max).exp())
                    .collect::<Vec<_>>();

                // Normalize them for each sample
                let sum = exp_values.iter().copied().sum::<f64>();
                exp_values
                    .into_iter()
                    .map(|value| value / sum)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }
}
