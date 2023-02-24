use std::ops::Div;

pub mod categorical_crossentropy;

pub trait Loss {
    fn forward(&self, predictions: &[Vec<f64>], actual: &[Vec<f64>]) -> Vec<f64>;

    fn calculate(&self, output: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
        // Calculate samples losses with the selected Loss function
        let sample_losses = self.forward(output, y);

        // Return mean loss
        sample_losses
            .iter()
            .sum::<f64>()
            .div(sample_losses.len() as f64)
    }
}
