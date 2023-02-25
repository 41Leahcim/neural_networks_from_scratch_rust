use std::ops::AddAssign;

use math::operations::{add, dot};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator, IndexedParallelIterator};

use crate::layer::Layer;

#[derive(Debug, Clone)]
pub struct Dense {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    outputs: Vec<Vec<f64>>,
}

impl Layer for Dense {
    /// Passes data through the layer, the values will be multiplied by the weights.
    /// The biases will be added to the result of those multiplications.
    /// Result is stored in the layer and retrieved with the ```get_outputs``` function.
    ///
    /// # Arguments
    /// ```inputs```: The inputs to process, output from the previous layer
    fn forward(&mut self, inputs: &[Vec<f64>]) {
        self.outputs = add::matrix_vector(&dot::matrix(inputs, &self.weights), &self.biases, false);
    }

    /// Returns a constant reference to the data.
    /// This will be an empty 2D Vector if the forward function hasn't been called yet.
    ///
    /// # Returns
    /// A constant reference to the data.
    fn get_outputs(&self) -> &[Vec<f64>] {
        &self.outputs
    }

    /// Adds a value to every weight
    fn add_matrix_to_weights(&mut self, matrix: &[Vec<f64>]){
        // Make sure the number of rows in the weights is equal to the number of rows in the matrix
        assert_eq!(self.weights.len(), matrix.len());
        self.weights.par_iter_mut().zip(matrix).for_each(|(weight_row, row)|{
            // Make sure the number of values in the weight row is equal to the number of values in the matrix row
            assert_eq!(weight_row.len(), row.len());

            // Multiply the values in the weight row with the values in the matrix row
            weight_row.iter_mut().zip(row).for_each(|(weight, value)| weight.add_assign(value));
        });
    }

    /// This function is not applicable for this funtion, as it doesn't have biases
    fn add_vector_to_biases(&mut self, vector: &[f64]){
        self.biases.iter_mut().zip(vector).for_each(|(bias, value)| bias.add_assign(value));
    }

    /// Returns the shape of the neural network
    fn shape(&self) -> (usize, usize){
        (self.weights.len(), self.biases.len())
    }
}

impl Dense {
    #[must_use]
    pub fn new(number_inputs: usize, number_neurons: usize) -> Self {
        let weights = math::randn_matrix(number_inputs, number_neurons);
        let biases = vec![0.0; number_neurons];

        Self {
            weights,
            biases,
            outputs: vec![vec![]],
        }
    }
}
