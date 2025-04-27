#![no_std]

pub mod activation;
#[cfg(feature = "dataset")]
pub mod dataset;
pub mod layer;
pub mod neuron;

#[cfg(test)]
const fn float_equal(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-8
}

#[cfg(test)]
mod tests {
    use rand::random;

    use crate::{
        activation::{relu::ReLu, softmax::Softmax},
        dataset::spiral,
        float_equal,
        layer::Dense,
    };

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
        let (x, _) = spiral(2, 3);

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
}
