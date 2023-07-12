use std::fmt::Debug;

use ndarray::Array2;

pub mod linear;
pub mod relu;
pub mod softmax;

pub trait Activation: Debug {
    /// Forward input through the layer, and stores the outputs
    fn forward(&mut self, inputs: Array2<f64>);

    /// Retrieves the outputs
    fn get_outputs(&self) -> &Array2<f64>;
}
