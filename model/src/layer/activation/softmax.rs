use crate::layer::Layer;

use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

#[derive(Debug, Clone, Default)]
pub struct Softmax {
    outputs: Vec<Vec<f64>>,
}

impl Layer for Softmax {
    /// Passes data through the layer, the values will be on a curve between 0 and 1.
    /// Result is stored in the layer and retrieved with the ```get_outputs``` function.
    ///
    /// # Arguments
    /// ```inputs```: The inputs to process, output from the previous layer
    fn forward(&mut self, inputs: &[Vec<f64>]) {
        self.outputs = inputs
            .par_iter()
            .map(|row| {
                let mut max = f64::NEG_INFINITY;
                row.iter().for_each(|value| max = value.max(max));
                let row = row
                    .iter()
                    .map(|value| (*value - max).exp())
                    .collect::<Vec<f64>>();
                let sum: f64 = row.iter().sum();
                row.iter().map(|value| *value / sum).collect()
            })
            .collect();
    }

    /// Returns a constant reference to the data.
    /// This will be an empty 2D Vector if the forward function hasn't been called yet.
    ///
    /// # Returns
    /// A constant reference to the data.
    fn get_outputs(&self) -> &[Vec<f64>] {
        &self.outputs
    }

    /// This function is not applicable for this funtion, as it doesn't have weights
    fn add_matrix_to_weights(&mut self, _: &[Vec<f64>]){}

    /// This function is not applicable for this funtion, as it doesn't have biases
    fn add_vector_to_biases(&mut self, _: &[f64]){}

    /// Returns the shape of the neural network
    fn shape(&self) -> (usize, usize){
        (0, 0)
    }
}
