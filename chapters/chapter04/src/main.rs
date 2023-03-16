#![warn(clippy::pedantic, clippy::nursery)]

use std::time::Instant;

use model::layer::{
    activation::{relu::ReLU, softmax::Softmax},
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
    let mut dense1 = Dense::new(2, 3);

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

    let mut layers: Vec<Box<dyn Layer>> = vec![
        Box::new(Dense::new(2, 3)), // Create a dense layer as input layer
        Box::<ReLU>::default(),           // Create a rectified Linear Activation funtion
        Box::new(Dense::new(3, 3)), // Create a dense layer as output layer
        Box::<Softmax>::default(),        // Create a Softmax Activation function
    ];

    // pass the input data in order through the layer
    layers[0].forward(&x);
    (1..layers.len()).for_each(|i| {
        let previous_outputs = layers[i - 1].get_outputs().to_owned();
        layers[i].forward(&previous_outputs);
    });

    // Print the first few results, if needed
    if PRINT_OUTPUT {
        println!("{}", &layers.last().unwrap().get_outputs().slice(s![..5, ..]));
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
