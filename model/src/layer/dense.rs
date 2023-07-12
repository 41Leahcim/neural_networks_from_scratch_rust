use std::ops::Add;

use ndarray::{Array, Array1, Array2};
use rand::Rng;

use super::{activation::Activation, Layer};

#[derive(Debug, Clone)]
pub struct Dense<const IN: usize, const OUT: usize, T: Activation> {
    weights: Array2<f64>,
    biases: Array1<f64>,
    activation: T,
}

impl<ActFunc: Activation, const IN: usize, const OUT: usize> Layer<ActFunc>
    for Dense<IN, OUT, ActFunc>
{
    /// Passes data through the layer, the values will be multiplied by the weights.
    /// The biases will be added to the result of those multiplications.
    /// Result is stored in the layer and retrieved with the ```get_outputs``` function.
    ///
    /// # Arguments
    /// ```inputs```: The inputs to process, output from the previous layer
    fn forward(&mut self, inputs: &Array2<f64>) {
        let outputs = inputs.dot(&self.weights).add(&self.biases);
        self.activation.forward(outputs);
    }

    /// Returns a constant reference to the data.
    /// This will be an empty 2D Vector if the forward function hasn't been called yet.
    ///
    /// # Returns
    /// A constant reference to the data.
    fn get_outputs(&self) -> &Array2<f64> {
        self.activation.get_outputs()
    }

    /// Returns the shape of the weights
    fn weights_shape(&self) -> &[usize] {
        self.weights.shape()
    }

    /// Returns the shape of the biases
    fn biases_shape(&self) -> &[usize] {
        self.biases.shape()
    }

    fn activation(&self) -> &ActFunc {
        &self.activation
    }
}

impl<T: Activation + Default, const IN: usize, const OUT: usize> Default for Dense<IN, OUT, T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<ActFunc: Activation, const IN: usize, const OUT: usize> Dense<IN, OUT, ActFunc> {
    #[must_use]
    pub fn new(activation: ActFunc) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array::from_shape_fn((IN, OUT), |_| rng.gen_range(-1.0..=1.0));
        let biases = Array1::zeros(OUT);

        Self {
            weights,
            biases,
            activation,
        }
    }
}
