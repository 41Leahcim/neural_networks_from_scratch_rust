#![warn(clippy::pedantic, clippy::nursery)]

use std::{
    ops::{Add, Sub},
    time::Instant,
};

use model::layer::{activation::linear::Linear, dense::Dense, Layer};
use ndarray::{array, s};

const PRINT_PERFORMANCE: bool = true;
const PRINT_OUTPUT: bool = false;

fn adding_layers() {
    // Start measuring performance
    let start = Instant::now();

    // Set the inputs
    let inputs = array![
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ];

    // Set the weights
    let weights = array![
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ];

    // Set the biases
    let bias = array![2.0, 3.0, 0.5];
    let weights2 = array![[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13],];

    // Set the biases
    let bias2 = array![-1.0, 2.0, -0.5];

    // Calculate the output of the first layer
    let layer1_outputs = inputs.dot(&weights.t()).add(bias);

    // Calculate the output of the second layer
    let layer2_outputs = layer1_outputs.dot(&weights2.t()).add(bias2);

    // Print the outputs if needed
    if PRINT_OUTPUT {
        println!("{layer2_outputs}");
    }

    // Print the performance if needed
    if PRINT_PERFORMANCE {
        println!("Adding layers: {}", start.elapsed().as_secs_f64());
    }

    // Check whether the result is valid
    array![
        [0.5031, -1.04185, -2.03875],
        [0.2434, -2.7332, -5.7633],
        [-0.99314, 1.41254, -0.35655],
    ]
    .iter()
    .zip(layer2_outputs)
    .for_each(|(expected, actual)| {
        let difference: f64 = expected.sub(actual);
        assert!(difference.abs().le(&0.000_000_000_000_001));
    });
}

fn dense_layer_class() {
    // Start measuring performance
    let start = Instant::now();

    // Generate input data
    let (x, _) = datasets::spiral(5_000_000, 3);

    // Create a dense layer
    let mut dense1 = Dense::<2, 3, Linear>::default();

    // Pass the data through the layer
    dense1.forward(&x);

    // Print the outputs if needed
    if PRINT_OUTPUT {
        println!("{}", &dense1.get_outputs().slice(s![..5, ..]));
    }

    // Print the performance if needed
    if PRINT_PERFORMANCE {
        println!("Dense layer class: {}", start.elapsed().as_secs_f64());
    }
}

fn main() {
    println!("Adding layers:");
    adding_layers();

    println!("Dense layer class:");
    dense_layer_class();
}
