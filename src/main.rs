use std::time::Instant;

use nnfs::{
    activation::{ReLU, Softmax},
    dataset, DenseLayer,
};

struct Network {
    dense1: DenseLayer<2, 3>,
    activation1: ReLU,
    dense2: DenseLayer<3, 3>,
    activation2: Softmax,
}

impl Network {
    pub fn new() -> Self {
        Self {
            dense1: DenseLayer::random(),
            activation1: ReLU,
            dense2: DenseLayer::random(),
            activation2: Softmax,
        }
    }

    pub fn forward(&self, inputs: Vec<[f64; 2]>) -> Vec<Vec<f64>> {
        let mut output = self.dense1.forward_batch(inputs);
        output = self.activation1.forward(output);
        output = self.dense2.forward_batch(output);
        self.activation2.forward(output)
    }
}

fn main() {
    let start = Instant::now();
    let data = dataset::spiral(10_000, 1_000);
    let network = Network::new();
    let output = network.forward(data.0);
    println!("{:?}", &output[..5]);
    assert_eq!(output.len(), 10_000_000);
    assert_eq!(output[0].len(), 3);
    println!("{:?}", start.elapsed());
}
