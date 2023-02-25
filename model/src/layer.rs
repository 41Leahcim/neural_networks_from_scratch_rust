pub mod activation;
pub mod dense;

use dyn_clone::DynClone;

pub trait Layer : DynClone {
    /// Forward input through the layer, and stores the outputs
    fn forward(&mut self, inputs: &[Vec<f64>]);

    /// Retrieves the outputs
    fn get_outputs(&self) -> &[Vec<f64>];

    /// Adds a matrix to the weights to train the network
    fn add_matrix_to_weights(&mut self, matrix: &[Vec<f64>]);

    /// Adds a matrix to the weights to train the network
    fn add_vector_to_biases(&mut self, vector: &[f64]);

    /// Returns the shape of the neural network
    fn shape(&self) -> (usize, usize);
}

dyn_clone::clone_trait_object!(Layer);
