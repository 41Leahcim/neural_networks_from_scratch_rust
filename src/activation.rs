use std::cmp::Ordering;

pub struct ReLU;

impl ReLU {
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

pub struct Softmax;

impl Softmax {
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
