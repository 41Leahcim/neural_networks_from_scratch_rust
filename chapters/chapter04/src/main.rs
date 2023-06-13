#![warn(clippy::pedantic, clippy::nursery)]

use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use model::layer::{
    activation::{relu::ReLU, softmax::Softmax, Activation},
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
    let mut dense1 = Dense::new(2, 3, None);

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
    let (x, _) = datasets::spiral(2_000_000, 3);

    let mut layers = vec![
        Dense::new(2, 3, Some(Arc::new(Mutex::new(ReLU::default())))), // Create a dense layer as input layer with a rectified Linear Activation funtion
        Dense::new(3, 3, Some(Arc::new(Mutex::new(Softmax::default())))), // Create a dense layer as output layer
    ];

    // pass the input data in order through the layer
    layers[0].forward(&x);
    (1..layers.len()).for_each(|i| {
        let previous_outputs = layers[i - 1].get_outputs().to_owned();
        layers[i].forward(&previous_outputs);
    });

    // Print the first few results, if needed
    if PRINT_OUTPUT {
        println!(
            "{}",
            &layers.last().unwrap().get_outputs().slice(s![..5, ..])
        );
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
