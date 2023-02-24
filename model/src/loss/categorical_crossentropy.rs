use std::ops::Mul;

use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use super::Loss;

#[derive(Debug, Default, Clone, Copy)]
pub struct CategoricalCrossentropy {}

impl Loss for CategoricalCrossentropy {
    fn forward(&self, predictions: &[Vec<f64>], actual: &[Vec<f64>]) -> Vec<f64> {
        // Make sure the matrices have the same size
        assert_eq!(predictions.len(), actual.len());

        // Clip data to prevent division by 0
        predictions
            .par_iter()
            .map(|row| {
                row.iter()
                    .map(|value| value.clamp(1e-7, 1.0 - 1e-7))
                    .collect::<Vec<f64>>()
            })
            .zip(actual)
            .map(|(predicted, actual)| {
                // Calculate loss (negative log of sum of the accuracies)
                -if actual.len() == 1 {
                    // If the expected value only contains 1 value, that value should be the label
                    // The prediction for that label is the accuracy
                    let actual = actual[0] as usize;
                    predicted[actual]
                } else {
                    // Make sure the vectors have the same size
                    assert_eq!(predicted.len(), actual.len());

                    // Sum the accuracies
                    predicted
                        .iter()
                        .zip(actual)
                        .map(|(predicted, actual)| {
                            // Multiply the prediction with the expected value
                            predicted.mul(actual)
                        })
                        .sum::<f64>()
                }
                .ln()
            })
            .collect::<Vec<f64>>()
    }
}
