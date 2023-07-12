use std::time::{Duration, Instant};

use model::{
    accuracy,
    layer::{
        activation::{relu::ReLU, softmax::Softmax},
        dense::Dense,
        Layer,
    },
    loss::{categorical_crossentropy::CategoricalCrossentropy, Loss},
};

const PRINT_LOSS: bool = true;
const PRINT_ACCURACY: bool = true;

fn categorical_crossentropy_test() {
    // Start measuring performance
    //let start = Instant::now();

    // Create the data
    let (x, y) = datasets::spiral(100, 3);

    // Create layers
    let mut layers = (
        Dense::<2, 3, ReLU>::default(), // Create a dense layer as input layer with a rectified Linear Activation funtion
        Dense::<3, 3, Softmax>::default(), // Create a dense layer as output layer
    );

    let loss_function = CategoricalCrossentropy::default();

    let mut best_loss = f64::MAX;

    let mut i: i128 = 0;
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(1) {
        // pass the input data in order through the layer
        layers.0.forward(&x);
        layers.1.forward(layers.0.outputs());

        // Create a loss function, and calculate loss
        let loss = loss_function.calculate(layers.1.outputs(), &y);

        if loss < best_loss {
            println!("Generation: {i}");
            best_loss = loss;

            // Print the loss, if needed
            if PRINT_LOSS {
                println!("Loss: {}", loss);
            }

            // Calculate and print the loss, if needed
            if PRINT_ACCURACY {
                println!("Accuracy: {}", accuracy(layers.1.outputs(), &y));
            }
            println!();
        }

        // Change the weights for the next iteration
        layers = (Dense::<2, 3, _>::default(), Dense::<3, 3, _>::default());
        i += 1;
    }
}

fn main() {
    println!("=== Categorical crossentropy test ===");
    categorical_crossentropy_test();
}
