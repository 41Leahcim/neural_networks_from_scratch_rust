use std::{
    ops::Add,
    sync::{Arc, Mutex},
};

use ndarray::{Array, Array1, Array2};
use rand::Rng;

use super::{activation::Activation, Layer};

#[derive(Debug, Clone)]
pub struct Dense {
    weights: Array2<f64>,
    biases: Array1<f64>,
    outputs: Array2<f64>,
    activation: Option<Arc<Mutex<dyn Activation>>>,
}

impl Layer for Dense {
    /// Passes data through the layer, the values will be multiplied by the weights.
    /// The biases will be added to the result of those multiplications.
    /// Result is stored in the layer and retrieved with the ```get_outputs``` function.
    ///
    /// # Arguments
    /// ```inputs```: The inputs to process, output from the previous layer
    fn forward(&mut self, inputs: &Array2<f64>) {
        self.outputs = inputs.dot(&self.weights).add(&self.biases);
        if let Some(activation) = self.activation.as_ref() {
            let mut activation = activation.lock().unwrap();
            activation.forward(&self.outputs);
            self.outputs = activation.get_outputs().clone();
        }
    }

    /// Returns a constant reference to the data.
    /// This will be an empty 2D Vector if the forward function hasn't been called yet.
    ///
    /// # Returns
    /// A constant reference to the data.
    fn get_outputs(&self) -> &Array2<f64> {
        &self.outputs
    }

    /// Returns the shape of the weights
    fn weights_shape(&self) -> &[usize] {
        self.weights.shape()
    }

    /// Returns the shape of the biases
    fn biases_shape(&self) -> &[usize] {
        self.biases.shape()
    }

    fn activation(&self) -> Option<Arc<Mutex<dyn Activation>>> {
        self.activation.clone()
    }
}

impl Dense {
    #[must_use]
    pub fn new(
        number_inputs: usize,
        number_neurons: usize,
        activation: Option<Arc<Mutex<dyn Activation>>>,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array::from_shape_fn((number_inputs, number_neurons), |_| {
            rng.gen_range(-1.0..=1.0)
        });
        let biases = Array1::zeros(number_neurons);

        Self {
            weights,
            biases,
            outputs: Array2::zeros((0, 0)),
            activation,
        }
    }
}
