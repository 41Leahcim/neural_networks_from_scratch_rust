#![warn(clippy::pedantic, clippy::nursery)]

use std::ops::Sub;

use numpy::operations::{self, add, dot};

fn a_single_neuron() {
    // Set the inputs, weights, and bias
    let inputs = vec![1.0, 2.0, 3.0];
    let weights = vec![0.2, 0.8, -0.5];
    let bias = 2.0;

    // Calculate the output
    let output = inputs
        .iter()
        .zip(weights)
        .map(|(&input, weight)| input * weight)
        .sum::<f64>()
        + bias;

    // Print and check the result
    println!("{output}");
    assert!(output.sub(2.3).abs().le(&f64::EPSILON));
}

fn a_larger_neuron() {
    // Set the inputs, weights, and bias
    let inputs = vec![1.0, 2.0, 3.0, 2.5];
    let weights = vec![0.2, 0.8, -0.5, 1.0];
    let bias = 2.0;

    // Calculate the output
    let output = inputs
        .iter()
        .zip(weights)
        .map(|(&input, weight)| input * weight)
        .sum::<f64>()
        + bias;

    // Print and check the result
    println!("{output}");
    assert!(output.sub(4.8).abs().le(&f64::EPSILON));
}

fn a_layer_of_neurons() {
    // Initialize the inputs, weights, and biases
    let inputs = vec![1.0, 2.0, 3.0, 2.5];
    let weights = vec![
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ];
    let biases = vec![2.0, 3.0, 0.5];

    // Calculate the outputs
    let output: Vec<f64> = weights
        .iter()
        .zip(biases)
        .map(|(weightset, bias)| -> f64 {
            weightset
                .iter()
                .zip(&inputs)
                .map(|(weight, input)| weight * input)
                .sum::<f64>()
                + bias
        })
        .collect();

    // Print and check the result
    println!("{output:?}");
    assert_eq!(output, vec![4.8, 1.21, 2.385]);
}

fn a_single_neuron_numpy() {
    // Initialize the inputs, weights, and bias
    let inputs = vec![1.0, 2.0, 3.0, 2.5];
    let weights = vec![0.2, 0.8, -0.5, 1.0];
    let bias = 2.0;

    // Calculate the output
    let output = dot::vectors(&inputs, &weights) + bias;

    // Print the result
    println!("{output}");
    assert!(output.sub(4.8).abs().le(&f64::EPSILON));
}

fn a_layer_of_neurons_numpy() {
    // Initialize the inputs, weights, and biases
    let inputs = vec![1.0, 2.0, 3.0, 2.5];
    let weights = vec![
        vec![0.2, 0.8, -0.5, 1.0],
        vec![0.5, -0.91, 0.26, -0.5],
        vec![-0.26, -0.27, 0.17, 0.87],
    ];
    let biases = vec![2.0, 3.0, 0.5];

    // Calculate the outputs
    let output = add::vectors(&dot::matrix_vector(&weights, &inputs), &biases);

    // Print and check the result
    println!("{output:?}");
    assert_eq!(output, vec![4.8, 1.21, 2.385]);
}

fn a_layer_of_neurons_and_batch_of_data_numpy() {
    // Initialize the inputs, weights, and biases
    let inputs = vec![
        vec![1.0, 2.0, 3.0, 2.5],
        vec![2.0, 5.0, -1.0, 2.0],
        vec![-1.5, 2.7, 3.3, -0.8],
    ];
    let weights = vec![
        vec![0.2, 0.8, -0.5, 1.0],
        vec![0.5, -0.91, 0.26, -0.5],
        vec![-0.26, -0.27, 0.17, 0.87],
    ];
    let biases = vec![2.0, 3.0, 0.5];

    // Calculate the outputs
    let output = add::matrix_vector(
        &dot::matrix(&inputs, &operations::t(&weights)),
        &biases,
        false,
    );

    // Print and check the outputs
    println!("{output:?}");
    vec![
        vec![4.8, 1.21, 2.385],
        vec![8.9, -1.81, 0.2],
        vec![1.41, 1.051, 0.026],
    ]
    .iter()
    .zip(output)
    .for_each(|(expected, actual)| {
        expected.iter().zip(actual).for_each(|(expected, actual)| {
            assert!(expected.sub(actual).abs().le(&1e-15));
        });
    });
}

fn main() {
    println!("A single neuron:");
    a_single_neuron();

    println!("\nA larger neuron:");
    a_larger_neuron();

    println!("\nA layer of neurons:");
    a_layer_of_neurons();

    println!("\nA single neuron numpy:");
    a_single_neuron_numpy();

    println!("\nA layer of neurons numpy:");
    a_layer_of_neurons_numpy();

    println!("\nA layer of neurons and a batch of data numpy:");
    a_layer_of_neurons_and_batch_of_data_numpy();
}
