//use std::time::Instant;

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use model::{
    accuracy,
    layer::{
        activation::{relu::ReLU, softmax::Softmax},
        dense::Dense,
        Layer,
    },
    loss::{categorical_crossentropy::CategoricalCrossentropy, Loss},
};

//const PRINT_OUTPUT: bool = false;
const PRINT_LOSS: bool = true;
const PRINT_ACCURACY: bool = true;
//const PRINT_PERFORMANCE: bool = true;

fn categorical_crossentropy_test() {
    // Start measuring performance
    //let start = Instant::now();

    // Create the data
    let (x, y) = datasets::spiral(100, 3);

    // Create layers
    let mut layers = vec![
        Dense::new(2, 3, Some(Arc::new(Mutex::new(ReLU::default())))), // Create a dense layer as input layer with a rectified Linear Activation funtion
        Dense::new(3, 3, Some(Arc::new(Mutex::new(Softmax::default())))), // Create a dense layer as output layer
    ];

    let loss_function = CategoricalCrossentropy::default();

    let mut best_loss = f64::MAX;
    let mut best_layers = vec![];

    let mut i: i128 = 0;
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(1) {
        // pass the input data in order through the layer
        layers[0].forward(&x);
        let mut previous_outputs = layers[0].get_outputs().to_owned();
        (1..layers.len()).for_each(|i| {
            layers[i].forward(&previous_outputs);
            previous_outputs = layers[i].get_outputs().to_owned();
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
        layers.iter_mut().step_by(2).for_each(|layer| {
            let shape = layer.weights_shape().to_owned();
            let activation = layer.activation();
            *layer = Dense::new(shape[0], shape[1], activation)
        });
        i += 1;
    }

    /*// Print the first few results, if needed
    if PRINT_OUTPUT {
        println!("{:?}", &layers.last().unwrap().get_outputs());
    }

    // Print the performance, if needed
    if PRINT_PERFORMANCE {
        println!("Run-time: {}", start.elapsed().as_secs_f64());
    }*/
}

fn main() {
    println!("=== Categorical crossentropy test ===");
    categorical_crossentropy_test();
}
