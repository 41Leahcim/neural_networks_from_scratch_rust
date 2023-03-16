#![warn(clippy::pedantic, clippy::nursery)]

use std::ops::{Sub, Add};

use ndarray::{array, Array1};

fn a_single_neuron() {
    // Set the inputs, weights, and bias
    let inputs = [1.0, 2.0, 3.0];
    let weights = [0.2, 0.8, -0.5];
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
    let inputs = [1.0, 2.0, 3.0, 2.5];
    let weights = [0.2, 0.8, -0.5, 1.0];
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
    let inputs = [1.0, 2.0, 3.0, 2.5];
    let weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ];
    let biases = [2.0, 3.0, 0.5];

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
    let inputs = array![1.0, 2.0, 3.0, 2.5];
    let weights = array![0.2, 0.8, -0.5, 1.0];
    let bias = 2.0;

    // Calculate the output
    let output = inputs.dot::<Array1<f64>>(&weights).add(bias);

    // Print the result
    println!("{output}");
    assert!(output.sub(4.8).abs().le(&f64::EPSILON));
}

fn a_layer_of_neurons_numpy() {
    // Initialize the inputs, weights, and biases
    let inputs = array![1.0, 2.0, 3.0, 2.5];
    let weights = array![
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ];
    let biases = array![2.0, 3.0, 0.5];

    // Calculate the outputs
    let output = weights.dot::<Array1<f64>>(&inputs).add(biases);

    // Print and check the result
    println!("{output}");
    assert_eq!(output, array![4.8, 1.21, 2.385]);
}

fn a_layer_of_neurons_and_batch_of_data_numpy() {
    // Initialize the inputs, weights, and biases
    let inputs = array![
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ];
    let weights = array![
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ];
    let biases = array![2.0, 3.0, 0.5];

    // Calculate the outputs
    let output = inputs.dot(&weights.t()).add(&biases);

    // Print and check the outputs
    println!("{output}");
    output.iter().zip(
        array![
        [4.8, 1.21, 2.385],
        [8.9, -1.81, 0.2],
        [1.41, 1.051, 0.026],
    ]
    ).for_each(|(output, expected)|{
        let difference: f64 = output.sub(expected);
        assert!(difference.abs() <= 0.000_000_000_000_001);
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
