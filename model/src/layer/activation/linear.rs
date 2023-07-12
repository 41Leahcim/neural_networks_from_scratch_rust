use ndarray::Array2;

use super::Activation;

#[derive(Debug, Clone, Default)]
pub struct Linear {
    outputs: Array2<f64>,
}

impl Activation for Linear {
    /// Passes data through the layer, setting negative value to 0.
    /// Result is stored in the layer and retrieved with the ```get_outputs``` function.
    ///
    /// # Arguments
    /// ```inputs```: The inputs to process, output from the previous layer
    fn forward(&mut self, inputs: Array2<f64>) {
        self.outputs = inputs;
    }

    /// Returns a constant reference to the data.
    /// This will be an empty 2D Vector if the forward function hasn't been called yet.
    ///
    /// # Returns
    /// A constant reference to the data.
    fn outputs(&self) -> &Array2<f64> {
        &self.outputs
    }
}
