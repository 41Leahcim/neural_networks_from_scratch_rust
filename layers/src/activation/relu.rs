use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::Layer;

#[derive(Debug, Clone, Default)]
pub struct ReLU {
    outputs: Vec<Vec<f64>>,
}

impl Layer for ReLU {
    /// Passes data through the layer, setting negative value to 0.
    /// Result is stored in the layer and retrieved with the ```get_outputs``` function.
    ///
    /// # Arguments
    /// ```inputs```: The inputs to process, output from the previous layer
    fn forward(&mut self, inputs: &[Vec<f64>]) {
        self.outputs = inputs
            .par_iter()
            .map(|vector| vector.iter().map(|value| value.max(0.0)).collect())
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
}
