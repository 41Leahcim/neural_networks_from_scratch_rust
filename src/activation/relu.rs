pub struct ReLu;

impl ReLu {
    pub const fn forward_sample<const SIZE: usize>(&self, sample: &[f64; SIZE]) -> [f64; SIZE] {
        let mut result = [0.0; SIZE];
        let mut i = 0;
        while i < SIZE {
            result[i] = sample[i].max(0.0);
            i += 1;
        }
        result
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
    use super::ReLu;

    #[test]
    fn test() {
        let inputs = [0.0, 2.0, -1.0, 3.3, -2.7, 1.1, 2.2, -100.0];
        let expected = [0.0, 2.0, 0.0, 3.3, 0.0, 1.1, 2.2, 0.0];
        assert!(
            ReLu.forward_sample(&inputs)
                .into_iter()
                .zip(expected)
                .all(|(left, right)| left == right)
        );
    }
}
