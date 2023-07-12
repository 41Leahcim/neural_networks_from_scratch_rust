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

const PRINT_OUTPUT: bool = false;
const PRINT_LOSS: bool = true;
const PRINT_ACCURACY: bool = true;
const PRINT_PERFORMANCE: bool = true;

fn categorical_crossentropy_test() {
    // Start measuring performance
    let start = Instant::now();

    // Create the data
    let (x, y) = datasets::spiral(1000000, 3);

    let mut layers = (
        Dense::<2, 3, ReLU>::default(), // Create a dense layer as input layer with a rectified Linear Activation funtion
        Dense::<3, 3, Softmax>::default(), // Create a dense layer as output layer with a a Softmax Activation function
    );

    // pass the input data in order through the layer
    layers.0.forward(&x);
    layers.1.forward(layers.0.outputs());

    // Create a loss function, and calculate loss
    let loss_function = CategoricalCrossentropy::default();
    let loss = loss_function.calculate(layers.1.outputs(), &y);

    // Print the first few results, if needed
    if PRINT_OUTPUT {
        println!("{:?}", &layers.1.outputs());
    }

    // Print the loss, if needed
    if PRINT_LOSS {
        println!("Loss: {}", loss);
    }

    // Calculate and print the loss, if needed
    if PRINT_ACCURACY {
        println!("Accuracy: {}", accuracy(layers.1.outputs(), &y));
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
