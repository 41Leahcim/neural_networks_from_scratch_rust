#![warn(clippy::pedantic, clippy::nursery)]
#![allow(clippy::must_use_candidate)]

use core::array;
use rand::Rng;
pub mod dataset;

pub struct Neuron<const INPUTS: usize> {
    weights: [f64; INPUTS],
    bias: f64,
}

impl<const INPUTS: usize> Neuron<INPUTS> {
    pub const fn new(weights: [f64; INPUTS], bias: f64) -> Self {
        Self { weights, bias }
    }

    pub fn forward(&self, inputs: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(inputs)
            .map(|(&weight, &input)| weight * input)
            .sum::<f64>()
            + self.bias
    }
}

pub struct DenseLayer<const INPUTS: usize, const OUTPUTS: usize> {
    neurons: [Neuron<INPUTS>; OUTPUTS],
}

impl<const INPUTS: usize, const OUTPUTS: usize> DenseLayer<INPUTS, OUTPUTS> {
    pub const fn new(neurons: [Neuron<INPUTS>; OUTPUTS]) -> Self {
        Self { neurons }
    }

    pub fn random() -> Self {
        let mut random = rand::thread_rng();
        Self {
            neurons: array::from_fn(|_| {
                Neuron::new(array::from_fn(|_| random.gen_range(-0.02..=0.02)), 0.0)
            }),
        }
    }

    pub fn forward_sample(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }

    pub fn forward_batch<Sample: AsRef<[f64]>>(&self, inputs: &[Sample]) -> Vec<Vec<f64>> {
        inputs
            .iter()
            .map(|input| self.forward_sample(input.as_ref()))
            .collect::<Vec<_>>()
    }
}
