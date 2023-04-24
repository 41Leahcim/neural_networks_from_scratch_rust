use std::ops::Add;

use ndarray::{Array2, Array1, Array};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Dense {
    weights: Array2<f64>,
    biases: Array1<f64>,
    outputs: Array2<f64>,
}

impl Dense {
    /// Passes data through the layer, the values will be multiplied by the weights.
    /// The biases will be added to the result of those multiplications.
    /// Result is stored in the layer and retrieved with the ```get_outputs``` function.
    ///
    /// # Arguments
    /// ```inputs```: The inputs to process, output from the previous layer
    pub fn forward(&mut self, inputs: &Array2<f64>) {
        self.outputs = inputs.dot(&self.weights).add(&self.biases);
    }

    /// Returns a constant reference to the data.
    /// This will be an empty 2D Vector if the forward function hasn't been called yet.
    ///
    /// # Returns
    /// A constant reference to the data.
    #[must_use]
    pub const fn get_outputs(&self) -> &Array2<f64> {
        &self.outputs
    }

    /// Returns the shape of the weights
    #[must_use]
    pub fn weights_shape(&self) -> [usize;2]{
        [self.weights.shape()[0], self.weights.shape()[1]]
    }
    
    /// Returns the shape of the biases
    #[must_use]
    pub fn biases_shape(&self) -> usize {
        self.biases.shape()[0]
    }
}

impl Dense {
    #[must_use]
    pub fn new(number_inputs: usize, number_neurons: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array::from_shape_fn((number_inputs, number_neurons), |_| rng.gen_range(-1.0..=1.0));
        let biases = Array1::zeros(number_neurons);

        Self {
            weights,
            biases,
            outputs: Array2::zeros((0, 0)),
        }
    }
}
