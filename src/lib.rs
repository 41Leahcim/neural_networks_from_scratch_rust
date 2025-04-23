#![no_std]

pub mod activation;
#[cfg(feature = "dataset")]
pub mod dataset;
pub mod layer;
pub mod neuron;

#[cfg(test)]
const fn float_equal(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-15
}
