use std::ops::Div;

use ndarray::{Array2, Array1};

pub mod categorical_crossentropy;

pub trait Loss {
    fn forward(&self, predictions: &Array2<f64>, actual: &Array2<f64>) -> Array1<f64>;

    fn calculate(&self, output: &Array2<f64>, y: &Array2<f64>) -> f64 {
        // Calculate samples losses with the selected Loss function
        let sample_losses = self.forward(output, y);

        // Return mean loss
        sample_losses
            .iter()
            .sum::<f64>()
            .div(sample_losses.len() as f64)
    }
}
