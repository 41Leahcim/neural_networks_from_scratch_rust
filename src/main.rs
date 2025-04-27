use neural_networks_from_scratch::{
    activation::{relu::ReLu, softmax::Softmax},
    dataset::vertical,
    layer::Dense,
    loss::{Loss, accuracy, categorical_crossentropy::CategoricalCrossentropy},
};
use rand::{random, random_range};

fn main() {
    let (x, y) = vertical(100, 3);
    let (x, y) = (
        x.collect::<Vec<_>>(),
        y.map(|value| [value]).collect::<Vec<_>>(),
    );

    // Create the model
    let mut dense1 = random::<Dense<2, 3>>();
    let activation1 = ReLu;
    let mut dense2 = random::<Dense<3, 3>>();
    let activation2 = Softmax;

    // Create loss function
    let loss_function = CategoricalCrossentropy;

    // Helper variables
    let mut lowest_loss = f64::INFINITY;
    let mut best_dense1 = dense1.clone();
    let mut best_dense2 = dense2.clone();

    for iteration in 0..1_000 {
        // Generate a new set of weights for iteration
        dense1.neurons.iter_mut().for_each(|neuron| {
            neuron
                .weights
                .iter_mut()
                .for_each(|weight| *weight += random_range(-0.05..=0.05));
            neuron.bias += random_range(-0.05..=0.05);
        });
        dense2.neurons.iter_mut().for_each(|neuron| {
            neuron
                .weights
                .iter_mut()
                .for_each(|weight| *weight += random_range(-0.05..=0.05));
            neuron.bias += random_range(-0.05..=0.05);
        });

        // Perform a forward pass of the training data through this layer
        let output = dense1.forward_batch(x.iter().copied());
        let output = activation1.forward_batch(output);
        let output = dense2.forward_batch(output);
        let output = activation2.forward_batch(output).collect::<Vec<_>>();

        // Perform a forward pass through activation function.
        // It takes the output of the second dense layer here and returns loss.
        let loss = loss_function.calculate(&output, y.iter().copied());

        // Calculate accuracy from output of activation2 and targets.
        let accuracy = accuracy(&output, y.iter().copied());

        // If loss is smaller, print and save weights and biases aside
        if loss < lowest_loss {
            println!(
                "New set of weights found, iteration: {iteration} loss: {loss} accuracy {accuracy}"
            );
            best_dense1 = dense1.clone();
            best_dense2 = dense2.clone();
            lowest_loss = loss;
        } else {
            dense1 = best_dense1.clone();
            dense2 = best_dense2.clone();
        }
    }
}
