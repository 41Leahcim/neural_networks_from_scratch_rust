pub mod categorical_crossentropy;

pub trait Loss<const SAMPLE_SIZE: usize, const LABEL_SIZE: usize> {
    /// Calculates the loss sum
    fn forward(
        &self,
        output: &[[f64; SAMPLE_SIZE]],
        y: impl Iterator<Item = [f64; LABEL_SIZE]>,
    ) -> f64;

    /// Calculates the data and regularization losses given the model output and ground truth
    /// values.
    fn calculate(
        &self,
        output: &[[f64; SAMPLE_SIZE]],
        y: impl Iterator<Item = [f64; LABEL_SIZE]>,
    ) -> f64 {
        // Calculate total loss
        let total_loss = self.forward(output, y);

        // Calculate and return mean loss
        total_loss / output.len() as f64
    }
}
