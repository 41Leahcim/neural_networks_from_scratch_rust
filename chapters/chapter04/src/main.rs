#![warn(clippy::pedantic, clippy::nursery)]

use std::time::Instant;

use model::layer::{
    activation::{linear::Linear, relu::ReLU, softmax::Softmax, Activation},
    dense::Dense,
    Layer,
};
use ndarray::s;

const PRINT_PERFORMANCE: bool = true;
const PRINT_OUTPUT: bool = false;

fn relu_test() {
    // Start measuring performance
    let start = Instant::now();

    // Create the data
    let (x, _) = datasets::spiral(4_000_000, 3);

    // Create a layer
    let mut dense1 = Dense::<2, 3, Linear>::default();

    // Create a ReLU (rectified linear) activation function
    let mut activation = ReLU::default();

    // Forward the data through the dense layer
    dense1.forward(&x);

    // Forward the output from the previous layer through the activation function
    activation.forward(dense1.get_outputs());

    // Print the first few results, if needed
    if PRINT_OUTPUT {
        println!("{}", &activation.get_outputs().slice(s![..5, ..]));
    }

    // Print the performance, if needed
    if PRINT_PERFORMANCE {
        println!("ReLU test: {}", start.elapsed().as_secs_f64());
    }
}

fn softmax_test() {
    // Start measuring performance
    let start = Instant::now();

    // Create the data
    let (x, _) = datasets::spiral(1_000_000, 3);

    let mut layers = (
        Dense::<2, 3, ReLU>::default(), // Create a dense layer as input layer with a rectified Linear Activation funtion
        Dense::<3, 3, Softmax>::default(), // Create a dense layer as output layer
    );

    // pass the input data in order through the layer
    layers.0.forward(&x);
    layers.1.forward(layers.0.get_outputs());

    // Print the first few results, if needed
    if PRINT_OUTPUT {
        println!("{}", &layers.1.get_outputs().slice(s![..5, ..]));
    }

    // Print the performance, if needed
    if PRINT_PERFORMANCE {
        println!("Softmax test: {}", start.elapsed().as_secs_f64());
    }
}

fn main() {
    eprintln!("=== ReLU test ===");
    relu_test();
    eprintln!("=== Softmax test ===");
    softmax_test();
}
