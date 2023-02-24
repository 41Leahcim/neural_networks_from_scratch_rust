pub mod activation;
pub mod dense;

pub trait Layer {
    fn forward(&mut self, inputs: &[Vec<f64>]);
    fn get_outputs(&self) -> &[Vec<f64>];
}
