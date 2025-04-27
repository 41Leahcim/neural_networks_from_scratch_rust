use core::array;

use super::Loss;

pub struct CategoricalCrossentropy;

impl<const SAMPLE_SIZE: usize, const LABEL_SIZE: usize> Loss<SAMPLE_SIZE, LABEL_SIZE>
    for CategoricalCrossentropy
{
    fn forward(
        &self,
        output: &[[f64; SAMPLE_SIZE]],
        y: impl Iterator<Item = [f64; LABEL_SIZE]>,
    ) -> f64 {
        // Clip data to prevent division by 0.
        // Clip both sides to not drag mean towards any value.
        let y_pred_clipped = output.iter().map(|sample| {
            array::from_fn::<f64, SAMPLE_SIZE, _>(|i| sample[i].clamp(1e-7, 1.0 - 1e-7))
        });

        // Calculate losses from probabilities
        if LABEL_SIZE == 1 {
            // Probabilities for target values - only if categorical labels
            y_pred_clipped
                .zip(y)
                .map(|(prediction, real)| -prediction[real[0].round() as usize].ln())
                .sum()
        } else if SAMPLE_SIZE == LABEL_SIZE {
            // Mask values - only for one-hot encoded labels
            y_pred_clipped
                .flatten()
                .zip(y.flatten())
                .map(|(prediction, real)| -(prediction * real).ln())
                .sum()
        } else {
            panic!(
                "Label size should be 1 or equal to sample size!\nSample size: {SAMPLE_SIZE}\nLabel size: {LABEL_SIZE}"
            );
        }
    }
}
