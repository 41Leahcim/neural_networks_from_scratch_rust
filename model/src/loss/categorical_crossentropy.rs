use std::ops::Mul;

use ndarray::{Array2, Array1};

use super::Loss;

#[derive(Debug, Default, Clone, Copy)]
pub struct CategoricalCrossentropy {}

impl Loss for CategoricalCrossentropy {
    fn forward(&self, predictions: &Array2<f64>, actual: &Array2<f64>) -> Array1<f64> {
        // Clip data to prevent division by 0
        predictions
            .mapv(|value| {
                value.clamp(1e-7, 1.0 - 1e-7)
            })
            .axis_iter(ndarray::Axis(0)).zip(actual.axis_iter(ndarray::Axis(0)))
            .map(|(predicted, actual)| {
                // Calculate loss (negative log of sum of the accuracies)
                -if actual.len() == 1 {
                    // If the expected value only contains 1 value, that value should be the label
                    // The prediction for that label is the accuracy
                    let actual = actual[0] as usize;
                    predicted[actual]
                } else {
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
            .collect::<Array1<f64>>()
    }
}
