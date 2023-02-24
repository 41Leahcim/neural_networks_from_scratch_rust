#![warn(clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::missing_panics_doc
)]
#![feature(sort_floats)]

use std::ops::Div;

pub mod layer;
pub mod loss;

#[must_use]
pub fn accuracy(predictions: &[Vec<f64>], actual: &[Vec<f64>]) -> f64 {
    // Make sure the prediction and actual value have the same outer size
    assert_eq!(predictions.len(), actual.len());
    let accuracies = predictions
        .iter()
        .zip(actual)
        .map(|(prediction, actual)| {
            if actual.len() == 1 {
                // If the expected value, is just one value
                // That value is the label, thus the prediction for that label is the accuracy
                prediction[actual[0] as usize]
            } else {
                // Make sure the inner-size is the same, if the expected value has more than one value
                assert_eq!(prediction.len(), actual.len());

                // Calculate the maximum, that is the main label
                let mut max = actual[0];
                let mut max_index = 0;
                actual.iter().enumerate().skip(1).for_each(|(i, value)| {
                    if max.le(value) {
                        max = *value;
                        max_index = i;
                    }
                });

                // Return the accuracy for that label
                prediction[max_index]
            }
        })
        .collect::<Vec<f64>>();

    // Return the average accuracy
    accuracies.iter().sum::<f64>().div(accuracies.len() as f64)
}
