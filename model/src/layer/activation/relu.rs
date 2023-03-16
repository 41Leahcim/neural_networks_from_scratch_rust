use ndarray::{Array2, Array1};

use crate::layer::Layer;

#[derive(Debug, Clone, Default)]
pub struct ReLU {
    outputs: Array2<f64>,
}

impl Layer for ReLU {
    /// Passes data through the layer, setting negative value to 0.
    /// Result is stored in the layer and retrieved with the ```get_outputs``` function.
    ///
    /// # Arguments
    /// ```inputs```: The inputs to process, output from the previous layer
    fn forward(&mut self, inputs: &Array2<f64>) {
        self.outputs = inputs.mapv(|value| value.max(0.0));
    }

    /// Returns a constant reference to the data.
    /// This will be an empty 2D Vector if the forward function hasn't been called yet.
    ///
    /// # Returns
    /// A constant reference to the data.
    fn get_outputs(&self) -> &Array2<f64> {
        &self.outputs
    }

    /// This function is not applicable for this funtion, as it doesn't have weights.
    fn add_matrix_to_weights(&mut self, _: &Array2<f64>) {}

    /// This function is not applicable for this function, as it doesn't have biases.
    fn add_vector_to_biases(&mut self, _: &Array1<f64>) {}

    /// Returns the shape of the weights
    fn weights_shape(&self) -> [usize;2]{
        [0, 0]
    }
    
    /// Returns the shape of the biases
    fn biases_shape(&self) -> usize {
        0
    }
}
