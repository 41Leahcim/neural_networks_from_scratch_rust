#![no_std]

pub mod activation;
#[cfg(feature = "dataset")]
pub mod dataset;
pub mod layer;
pub mod loss;
pub mod neuron;

const fn float_equal(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-5
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use crate::{
        activation::{relu::ReLu, softmax::Softmax},
        dataset::spiral,
        float_equal,
        layer::Dense,
        loss::{Loss, accuracy, categorical_crossentropy::CategoricalCrossentropy},
    };
    use alloc::vec::Vec;
    use rand::random;

    #[test]
    fn dense_layer_with_relu() {
        let (x, _) = spiral(100, 3);
        let dense = random::<Dense<2, 3>>();
        let mut output = ReLu.forward_batch(dense.forward_batch(x));
        assert!(output.next().unwrap().into_iter().all(|value| value == 0.0));
        assert!(output.flatten().all(|value| value >= 0.0));
    }

    #[test]
    fn dense_layer_with_activation_functions() {
        // Create dataset
        let (x, _) = spiral(100, 3);

        // Create a dense layer
        let dense1 = random::<Dense<2, 3>>();

        // Create a ReLU activation layer
        let activation1 = ReLu;

        // Create the second dense layer with 3 input features (as it takes the output of the
        // previous layer here) and 3 output values
        let dense2 = random::<Dense<3, 3>>();

        // Create the Softmax activation (to be used with dense2)
        let activation2 = Softmax;
        let output = dense1.forward_batch(x);
        let output = activation1.forward_batch(output);
        let output = dense2.forward_batch(output);
        let output = activation2.forward_batch(output);

        assert!(output.flatten().all(|value| float_equal(value, 1.0 / 3.0)));
    }

    #[test]
    fn model_with_categorical_loss() {
        const SAMPLES: usize = 100;
        const CLASSES: usize = 3;
        // Create dataset
        let (x, y) = spiral(SAMPLES, CLASSES);

        // Create a dense layer
        let dense1 = random::<Dense<2, 3>>();

        // Create a ReLU activation layer
        let activation1 = ReLu;

        // Create the second dense layer with 3 input features (as it takes the output of the
        // previous layer here) and 3 output values
        let dense2 = random::<Dense<3, 3>>();

        // Create the Softmax activation (to be used with dense2)
        let activation2 = Softmax;

        // Create loss function
        let loss_function = CategoricalCrossentropy;

        let output = dense1.forward_batch(x);
        let output = activation1.forward_batch(output);
        let output = dense2.forward_batch(output);
        let output = activation2.forward_batch(output);
        let output = output.collect::<Vec<_>>();
        let y = y.map(|value| [value]).collect::<Vec<_>>();
        assert!(float_equal(
            loss_function.calculate(&output, y.iter().copied()),
            1.0986104
        ));
        assert!(float_equal(accuracy(&output, y.iter().copied()), 1.0 / 3.0));
    }
}
