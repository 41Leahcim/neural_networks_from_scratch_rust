#![warn(clippy::pedantic, clippy::nursery)]
#![allow(clippy::must_use_candidate)]

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

pub struct Layer<const INPUTS: usize, const OUTPUTS: usize> {
    neurons: [Neuron<INPUTS>; OUTPUTS],
}

impl<const INPUTS: usize, const OUTPUTS: usize> Layer<INPUTS, OUTPUTS> {
    pub const fn new(neurons: [Neuron<INPUTS>; OUTPUTS]) -> Self {
        Self { neurons }
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
