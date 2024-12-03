struct Neuron<const INPUTS: usize> {
    weights: [f64; INPUTS],
    bias: f64,
}

impl<const INPUTS: usize> Neuron<INPUTS> {
    fn forward(&self, inputs: &[f64]) -> f64 {
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
    fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }
}

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
            [8.9, -1.8099999999999996, 0.19999999999999996],
            [1.4100000000000001, 1.0509999999999997, 0.025999999999999912]
        ]
    );
}

fn main() {
    single_neuron();
    neuron_layer();
    neuron_layer_with_batch_of_data();
}
