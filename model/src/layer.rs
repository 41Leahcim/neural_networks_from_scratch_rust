pub mod activation;
pub mod dense;

use ndarray::Array2;

use self::activation::softmax::Softmax;
use self::dense::Dense;
use self::activation::relu::ReLU;

#[derive(Debug, Clone)]
pub enum Layer {
    Dense(Dense),
    ReLU(ReLU),
    Softmax(Softmax)
}

impl Layer{
    /// Forward input through the layer, and stores the outputs
    pub fn forward(&mut self, inputs: &Array2<f64>){
        match self {
            Self::Dense(layer) => layer.forward(inputs),
            Self::ReLU(layer) => layer.forward(inputs),
            Self::Softmax(layer) => layer.forward(inputs)
        }
    }

    /// Retrieves the outputs
    #[must_use]
    pub const fn get_outputs(&self) -> &Array2<f64>{
        match self {
            Self::Dense(layer) => layer.get_outputs(),
            Self::ReLU(layer) => layer.get_outputs(),
            Self::Softmax(layer) => layer.get_outputs()
        }
    }

    /// Returns the shape of the weights
    #[must_use]
    pub fn weights_shape(&self) -> Option<[usize;2]>{
        match self {
            Self::Dense(layer) => Some(layer.weights_shape()),
            Self::ReLU(_) | Self::Softmax(_) => None,
        }
    }
    
    /// Returns the shape of the biases
    #[must_use]
    pub fn biases_shape(&self) -> Option<usize>{
        match self {
            Self::Dense(layer) => Some(layer.biases_shape()),
            Self::ReLU(_) | Self::Softmax(_) => None,
        }
    }
}
