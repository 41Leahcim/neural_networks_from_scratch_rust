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

#[cfg(test)]
mod tests {
    use rand::random;

    use crate::{activation::relu::ReLu, dataset::spiral, layer::Dense};

    #[test]
    fn dense_layer_with_activation() {
        let (x, _) = spiral(100, 3);
        let dense = random::<Dense<2, 3>>();
        let mut output = ReLu.forward_batch(dense.forward_batch(x));
        assert!(output.next().unwrap().into_iter().all(|value| value == 0.0));
        assert!(output.flatten().all(|value| value >= 0.0));
    }
}
