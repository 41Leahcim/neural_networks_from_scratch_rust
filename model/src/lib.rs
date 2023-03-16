#![warn(clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::missing_panics_doc
)]
#![feature(sort_floats)]

use std::ops::Div;

use ndarray::Array2;

pub mod layer;
pub mod loss;

#[must_use]
pub fn accuracy(predictions: &Array2<f64>, actual: &Array2<f64>) -> f64 {
    // Get the number of samples
    let samples = predictions.shape()[0];

    predictions
        .axis_iter(ndarray::Axis(0))
        .zip(actual.axis_iter(ndarray::Axis(0)))
        .map(|(prediction, actual)| {
            if actual.len() == 1 {
                // If the expected value, is just one value
                // That value is the label, thus the prediction for that label is the accuracy
                prediction[actual[0] as usize]
            } else {
                // Make sure the vectors have the same size
                assert_eq!(prediction.len(), actual.len());

                // Calculate the maximum, that is the main label
                let mut max = actual[0];
                let mut max_index = 0;
                actual.iter().enumerate().skip(1).for_each(|(i, value)| {
                    if value.gt(&max) {
                        max = *value;
                        max_index = i;
                    }
                });

                // Return the accuracy for that label
                prediction[max_index]
            }
        })
        .sum::<f64>().div(samples as f64)
}
