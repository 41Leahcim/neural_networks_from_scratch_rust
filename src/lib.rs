//! A Rust version of the main code of neural networks from scratch.

#![warn(clippy::pedantic, clippy::nursery, missing_docs)]
#![allow(clippy::must_use_candidate)]

use core::array;
use rand::Rng;

pub mod activation;
pub mod dataset;

/// A simple neuron.
/// Weights are multiplied with the respective input.
/// Those values are added with the bias and returned.
pub struct Neuron<const INPUTS: usize> {
    weights: [f64; INPUTS],
    bias: f64,
}

impl<const INPUTS: usize> Neuron<INPUTS> {
    /// Create a new neuron with at compile time known values.
    pub const fn new(weights: [f64; INPUTS], bias: f64) -> Self {
        Self { weights, bias }
    }

    /// Forwards the data through the neuron.
    /// The output will be the result of multiplying the weights with the respective inputs,
    /// summing those results and adding the bias.
    pub fn forward<Input: AsRef<[f64]>>(&self, inputs: Input) -> f64 {
        self.weights
            .iter()
            .zip(inputs.as_ref())
            .map(|(&weight, &input)| weight * input)
            .sum::<f64>()
            + self.bias
    }
}

/// A layer consisting of `OUTPUT` neurons, each having `INPUTS` weights.
pub struct DenseLayer<const INPUTS: usize, const OUTPUTS: usize> {
    neurons: [Neuron<INPUTS>; OUTPUTS],
}

impl<const INPUTS: usize, const OUTPUTS: usize> DenseLayer<INPUTS, OUTPUTS> {
    /// Creates a layer with at compile-time known neurons.
    pub const fn new(neurons: [Neuron<INPUTS>; OUTPUTS]) -> Self {
        Self { neurons }
    }

    /// Creates a layer with neurons initialized with random weights, biases set to 0.
    pub fn random() -> Self {
        let mut random = rand::thread_rng();
        Self {
            neurons: array::from_fn(|_| {
                Neuron::new(array::from_fn(|_| random.gen_range(-0.02..=0.02)), 0.0)
            }),
        }
    }

    /// Forwards a single sample through all neurons in the layer.
    pub fn forward_sample<Sample: AsRef<[f64]> + Copy>(&self, inputs: Sample) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }

    /// Forwards multiple samples through all neurons in the layer.
    /// The size of the outer vector will stay the same, the inner size will become `OUTPUT`.
    pub fn forward_batch<Sample: AsRef<[f64]>, Input: IntoIterator<Item = Sample>>(
        &self,
        inputs: Input,
    ) -> Vec<Vec<f64>> {
        inputs
            .into_iter()
            .map(|input| self.forward_sample(input.as_ref()))
            .collect::<Vec<_>>()
    }
}
