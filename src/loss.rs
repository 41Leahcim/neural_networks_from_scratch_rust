use crate::float_equal;

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

fn arg_max(iter: impl Iterator<Item = f64>) -> usize {
    iter.enumerate()
        .max_by(|(_, value), (_, value2)| value.total_cmp(value2))
        .map(|(index, _)| index)
        .unwrap()
}

pub fn accuracy<const SAMPLE_SIZE: usize, const LABEL_SIZE: usize>(
    output: &[[f64; SAMPLE_SIZE]],
    y: impl Iterator<Item = [f64; LABEL_SIZE]>,
) -> f64 {
    // Calculate accuracy from output and targets.
    // Calculate values along first axis
    let predictions = output.iter().map(|sample| arg_max(sample.iter().copied()));

    (if LABEL_SIZE == SAMPLE_SIZE {
        y.zip(predictions)
            .filter(|(real, prediction)| arg_max(real.iter().copied()) == *prediction)
            .count() as f64
    } else if LABEL_SIZE == 1 {
        y.flatten()
            .zip(predictions)
            .filter(|&(real, prediction)| float_equal(prediction as f64, real))
            .count() as f64
    } else {
        panic!(
            "Label size must be 1 or equal to sample size!\nSample size: {SAMPLE_SIZE}\nLabel size: {LABEL_SIZE}"
        );
    } / output.len() as f64)
}
