use std::time::Instant;

use model::{
    accuracy,
    layer::{
        activation::{relu::ReLU, softmax::Softmax},
        dense::Dense,
        Layer,
    },
    loss::{categorical_crossentropy::CategoricalCrossentropy, Loss},
};
use ndarray::Array2;

const PRINT_OUTPUT: bool = false;
const PRINT_LOSS: bool = true;
const PRINT_ACCURACY: bool = true;
const PRINT_PERFORMANCE: bool = true;

fn categorical_crossentropy_test() {
    // Start measuring performance
    let start = Instant::now();

    // Create the data
    let (x, y) = datasets::spiral(100, 3);

    // Create layers
    let mut layers: Vec<Box<dyn Layer>> = vec![
        Box::new(Dense::new(2, 3)), // Create a dense layer as input layer
        Box::<ReLU>::default(),     // Create a rectified Linear Activation funtion
        Box::new(Dense::new(3, 3)), // Create a dense layer as output layer
        Box::<Softmax>::default(),  // Create a Softmax Activation function
    ];

    let loss_function = CategoricalCrossentropy::default();

    let mut best_loss = f64::MAX;
    let mut best_layers = vec![];

    for i in 0..100_000 {
        // pass the input data in order through the layer
        layers[0].forward(&x);
        (1..layers.len()).for_each(|i| {
            let previous_outputs = layers[i - 1].get_outputs().to_owned();
            layers[i].forward(&previous_outputs);
        });

        // Create a loss function, and calculate loss
        let loss = loss_function.calculate(layers.last().unwrap().get_outputs(), &y);

        if loss < best_loss {
            println!("Generation: {i}");
            best_loss = loss;
            best_layers = layers.to_vec();

            // Print the loss, if needed
            if PRINT_LOSS {
                println!("Loss: {}", loss);
            }

            // Calculate and print the loss, if needed
            if PRINT_ACCURACY {
                println!(
                    "Accuracy: {}",
                    accuracy(layers.last().unwrap().get_outputs(), &y)
                );
            }
            println!();
        } else {
            layers = best_layers.to_vec();
        }

        // Change the weights for the next iteration
        const LEARNING_RATE: f64 = 0.05;
        layers.iter_mut().step_by(2).for_each(|layer| {
            let matrix = Array2::from_shape_fn(layer.weights_shape(), |_| rand::random::<f64>() * 2.0 * LEARNING_RATE - 1.0);
            layer.add_matrix_to_weights(&matrix);
        });
    }

    // Print the first few results, if needed
    if PRINT_OUTPUT {
        println!("{:?}", &layers.last().unwrap().get_outputs());
    }

    // Print the performance, if needed
    if PRINT_PERFORMANCE {
        println!("Run-time: {}", start.elapsed().as_secs_f64());
    }
}

fn main() {
    println!("=== Categorical crossentropy test ===");
    categorical_crossentropy_test();
}
