use crate::layer::Layer;

use ndarray::{Array2, Axis};

#[derive(Debug, Clone, Default)]
pub struct Softmax {
    outputs: Array2<f64>,
}

impl Layer for Softmax {
    /// Passes data through the layer, the values will be on a curve between 0 and 1.
    /// Result is stored in the layer and retrieved with the ```get_outputs``` function.
    ///
    /// # Arguments
    /// ```inputs```: The inputs to process, output from the previous layer
    fn forward(&mut self, inputs: &Array2<f64>) {
        let output_vec = inputs
            .axis_iter(Axis(0))
            .map(|row| {
                let max = row.fold(f64::NEG_INFINITY, |acc, &value| value.max(acc));
                let row_exp = row.map(|&value| (value - max).exp());
                let sum: f64 = row_exp.sum();
                (row_exp / sum).to_owned()
            })
            .collect::<Vec<_>>();
        
        let num_rows = output_vec.len();
        let num_cols = output_vec[0].len();
        self.outputs = Array2::zeros((num_rows, num_cols));

        output_vec.iter().enumerate().for_each(|(i, row)|{
            self.outputs.row_mut(i).assign(row);
        });

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
    fn weights_shape(&self) -> [usize;2]{
        [0, 0]
    }
    
    /// Returns the shape of the biases
    fn biases_shape(&self) -> usize {
        0
    }
}
