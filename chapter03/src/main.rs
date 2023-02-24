#![warn(clippy::pedantic, clippy::nursery)]

use std::{ops::Sub, time::Instant};

use layers::{dense::Dense, Layer};
use numpy::operations::{self, add, dot};

const PRINT_PERFORMANCE: bool = false;
const PRINT_OUTPUT: bool = true;

fn adding_layers() {
    // Start measuring performance
    let start = Instant::now();

    // Set the inputs
    let inputs = vec![
        vec![1.0, 2.0, 3.0, 2.5],
        vec![2.0, 5.0, -1.0, 2.0],
        vec![-1.5, 2.7, 3.3, -0.8],
    ];

    // Set the weights
    let weights = vec![
        vec![0.2, 0.8, -0.5, 1.0],
        vec![0.5, -0.91, 0.26, -0.5],
        vec![-0.26, -0.27, 0.17, 0.87],
    ];

    // Set the biases
    let bias = vec![2.0, 3.0, 0.5];
    let weights2 = vec![
        vec![0.1, -0.14, 0.5],
        vec![-0.5, 0.12, -0.33],
        vec![-0.44, 0.73, -0.13],
    ];

    // Set the biases
    let bias2 = vec![-1.0, 2.0, -0.5];

    // Calculate the output of the first layer
    let layer1_outputs = add::matrix_vector(
        &dot::matrix(&inputs, &operations::t(&weights)),
        &bias,
        false,
    );

    // Calculate the output of the second layer
    let layer2_outputs = add::matrix_vector(
        &dot::matrix(&layer1_outputs, &operations::t(&weights2)),
        &bias2,
        false,
    );

    // Print the outputs if needed
    if PRINT_OUTPUT {
        println!("{layer2_outputs:?}");
    }

    // Print the performance if needed
    if PRINT_PERFORMANCE {
        println!("Adding layers: {}", start.elapsed().as_secs_f64());
    }

    // Check whether the result is valid
    vec![
        vec![0.5031, -1.04185, -2.03875],
        vec![0.2434, -2.7332, -5.7633],
        vec![-0.99314, 1.41254, -0.35655],
    ]
    .iter()
    .zip(layer2_outputs)
    .for_each(|(expected, actual)| {
        expected.iter().zip(actual).for_each(|(expected, actual)| {
            assert!(expected.sub(actual).abs().le(&0.000_000_000_000_001));
        });
    });
}

fn dense_layer_class() {
    // Start measuring performance
    let start = Instant::now();

    // Generate input data
    let (x, _) = datasets::spiral(1_000_000, 3);

    // Create a dense layer
    let mut dense1 = Dense::new(2, 3);

    // Pass the data through the layer
    dense1.forward(&x);

    // Print the outputs if needed
    if PRINT_OUTPUT {
        println!("{:?}", &dense1.get_outputs()[0..5]);
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
