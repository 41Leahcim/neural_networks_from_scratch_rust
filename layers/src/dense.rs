use math::operations::{add, dot};

use crate::Layer;

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
