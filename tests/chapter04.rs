use nnfs::{
    activation::{ReLU, Softmax},
    dataset, DenseLayer,
};

#[test]
fn relu() {
    let data = dataset::spiral(1_000, 1_000);
    let dense = DenseLayer::<2, 3>::random();
    let mut output = dense.forward_batch(&data.0);
    let activation = ReLU;
    output = activation.forward(output);
    assert_eq!(output.len(), 10_000_000);
    assert_eq!(output[0].len(), 3);
}

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

#[test]
fn softmax() {
    let data = dataset::spiral(1_000, 1_000);
    let network = Network::new();
    let output = network.forward(data.0);
    assert_eq!(output.len(), 1_000_000);
    assert_eq!(output[0].len(), 3);
}
