use core::array;

use rand::distr::{Distribution, StandardUniform};

#[derive(Debug, Clone)]
pub struct Neuron<const SIZE: usize> {
    pub weights: [f64; SIZE],
    pub bias: f64,
}

impl<const SIZE: usize> Neuron<SIZE> {
    pub const fn new(weights: [f64; SIZE], bias: f64) -> Self {
        Self { weights, bias }
    }

    pub const fn forward(&self, inputs: &[f64; SIZE]) -> f64 {
        let mut result = self.bias;
        let mut i = 0;
        while i < inputs.len() {
            result += self.weights[i] * inputs[i];
            i += 1;
        }
        result
    }
}

#[cfg(feature = "rand")]
impl<const SIZE: usize> Distribution<Neuron<SIZE>> for StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Neuron<SIZE> {
        Neuron {
            weights: array::from_fn(|_| rng.random_range(0.01..=0.01)),
            bias: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::float_equal;

    use super::Neuron;

    #[test]
    fn test1() {
        const INPUTS: [f64; 3] = [1.0, 2.0, 3.0];
        const NEURON: Neuron<3> = Neuron::new([0.2, 0.8, -0.5], 2.0);
        const OUTPUT: f64 = NEURON.forward(&INPUTS);
        assert!(float_equal(OUTPUT, 2.3));
    }

    #[test]
    fn test2() {
        const INPUTS: [f64; 4] = [1.0, 2.0, 3.0, 2.5];
        const NEURON: Neuron<4> = Neuron::new([0.2, 0.8, -0.5, 1.0], 2.0);
        const OUTPUT: f64 = NEURON.forward(&INPUTS);
        assert!(float_equal(OUTPUT, 4.8));
    }
}
