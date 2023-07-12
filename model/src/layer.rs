pub mod activation;
pub mod dense;

use std::fmt::Debug;

use ndarray::Array2;

use self::activation::Activation;

pub trait Layer<ActFunc: Activation>: Debug {
    /// Forward input through the layer, and stores the outputs
    fn forward(&mut self, inputs: &Array2<f64>);

    /// Retrieves the outputs
    fn get_outputs(&self) -> &Array2<f64>;

    /// Returns the shape of the weights
    fn weights_shape(&self) -> &[usize];

    /// Returns the shape of the biases
    fn biases_shape(&self) -> &[usize];

    // Return activation layer
    fn activation(&self) -> &ActFunc;
}
