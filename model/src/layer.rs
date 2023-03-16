pub mod activation;
pub mod dense;

use dyn_clone::DynClone;
use ndarray::Array2;

pub trait Layer: DynClone {
    /// Forward input through the layer, and stores the outputs
    fn forward(&mut self, inputs: &Array2<f64>);

    /// Retrieves the outputs
    fn get_outputs(&self) -> &Array2<f64>;

    /// Returns the shape of the weights
    fn weights_shape(&self) -> [usize;2];
    
    /// Returns the shape of the biases
    fn biases_shape(&self) -> usize;
}

dyn_clone::clone_trait_object!(Layer);
