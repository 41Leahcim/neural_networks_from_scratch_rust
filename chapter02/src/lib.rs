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

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }
}

#[test]
fn single_neuron() {
    let inputs = [1.0, 2.0, 3.0];
    let neuron = Neuron {
        weights: [0.2, 0.8, -0.5],
        bias: 2.0,
    };
    let output = neuron.forward(&inputs);
    assert!((output - 2.3).abs() < f64::EPSILON);
    println!("{output}");
}

#[test]
fn neuron_layer() {
    let inputs = [1.0, 2.0, 3.0, 2.5];
    let neurons = Layer {
        neurons: [
            Neuron {
                weights: [0.2, 0.8, -0.5, 1.0],
                bias: 2.0,
            },
            Neuron {
                weights: [0.5, -0.91, 0.26, -0.5],
                bias: 3.0,
            },
            Neuron {
                weights: [-0.26, -0.27, 0.17, 0.87],
                bias: 0.5,
            },
        ],
    };
    let output = neurons.forward(&inputs);
    println!("{output:?}");
    assert_eq!(output, [4.8, 1.21, 2.385]);
}

#[test]
fn neuron_layer_with_batch_of_data() {
    let inputs = [
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ];
    let neurons = Layer {
        neurons: [
            Neuron {
                weights: [0.2, 0.8, -0.5, 1.0],
                bias: 2.0,
            },
            Neuron {
                weights: [0.5, -0.91, 0.26, -0.5],
                bias: 3.0,
            },
            Neuron {
                weights: [-0.26, -0.27, 0.17, 0.87],
                bias: 0.5,
            },
        ],
    };
    let outputs = inputs
        .iter()
        .map(|input| neurons.forward(input))
        .collect::<Vec<_>>();
    println!("{outputs:?}");
    assert_eq!(
        outputs,
        [
            [4.8, 1.21, 2.385],
            [8.9, -1.809_999_999_999_999_6, 0.199_999_999_999_999_96],
            [
                1.410_000_000_000_000_1,
                1.050_999_999_999_999_7,
                0.025_999_999_999_999_912
            ]
        ]
    );
}
