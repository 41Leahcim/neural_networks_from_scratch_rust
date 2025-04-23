use core::array;

pub struct Softmax;

impl Softmax {
    pub fn forward_sample<const SIZE: usize>(&self, sample: &[f64; SIZE]) -> [f64; SIZE] {
        let exp_values: [f64; SIZE] = array::from_fn(|i| sample[i].exp());
        eprintln!("{exp_values:?}");
        let sum = exp_values.iter().sum::<f64>();
        eprintln!("{sum}");
        array::from_fn(|i| exp_values[i] / sum)
    }

    pub fn forward_batch<const SIZE: usize>(
        &self,
        batch: impl IntoIterator<Item = [f64; SIZE]>,
    ) -> impl Iterator<Item = [f64; SIZE]> {
        batch.into_iter().map(|sample| self.forward_sample(&sample))
    }
}

#[cfg(test)]
mod tests {
    use super::Softmax;
    use crate::float_equal;

    #[test]
    fn test() {
        let layer_outputs = [4.8, 1.21, 2.385];
        let norm_values = Softmax.forward_sample(&layer_outputs);
        assert!(
            norm_values
                .into_iter()
                .zip([0.89528266, 0.02470831, 0.08000903])
                .all(|(left, right)| float_equal(left, right))
        );
        assert!(float_equal(norm_values.into_iter().sum::<f64>(), 1.0));
    }
}
