#[cfg(feature = "rand")]
use core::array;

#[cfg(feature = "rand")]
use rand::distr::{Distribution, StandardUniform};

use crate::neuron::Neuron;

#[derive(Debug, Clone)]
pub struct Dense<const INPUT: usize, const OUTPUT: usize> {
    pub neurons: [Neuron<INPUT>; OUTPUT],
}

impl<const INPUT: usize, const OUTPUT: usize> Dense<INPUT, OUTPUT> {
    pub const fn new(neurons: [Neuron<INPUT>; OUTPUT]) -> Self {
        Self { neurons }
    }

    pub const fn forward_sample(&self, input: &[f64; INPUT]) -> [f64; OUTPUT] {
        let mut outputs = [0.0; OUTPUT];
        let mut i = 0;
        while i < OUTPUT {
            outputs[i] = self.neurons[i].forward(input);
            i += 1;
        }
        outputs
    }

    pub fn forward_batch<Iter: IntoIterator<Item = [f64; INPUT]>>(
        &self,
        batch: Iter,
    ) -> impl Iterator<Item = [f64; OUTPUT]> {
        batch.into_iter().map(|sample| self.forward_sample(&sample))
    }
}

#[cfg(feature = "rand")]
impl<const INPUT: usize, const OUTPUT: usize> Distribution<Dense<INPUT, OUTPUT>>
    for StandardUniform
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Dense<INPUT, OUTPUT> {
        Dense::new(array::from_fn(|_| rng.random()))
    }
}

#[cfg(test)]
mod tests {
    use rand::random;

    use crate::{dataset::spiral, float_equal, neuron::Neuron};

    use super::Dense;

    #[test]
    fn layer_with_sample() {
        const INPUTS: [f64; 4] = [1.0, 2.0, 3.0, 2.5];
        const LAYER: Dense<4, 3> = Dense::new([
            Neuron::new([0.2, 0.8, -0.5, 1.0], 2.0),
            Neuron::new([0.5, -0.91, 0.26, -0.5], 3.0),
            Neuron::new([-0.26, -0.27, 0.17, 0.87], 0.5),
        ]);
        const OUTPUTS: [f64; 3] = LAYER.forward_sample(&INPUTS);
        assert!(
            OUTPUTS
                .into_iter()
                .zip([4.8, 1.21, 2.385])
                .all(|(left, right)| float_equal(left, right))
        );
    }

    #[test]
    fn layer_with_batch() {
        const INPUTS: [[f64; 4]; 3] = [
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8],
        ];
        const LAYER: Dense<4, 3> = Dense::new([
            Neuron::new([0.2, 0.8, -0.5, 1.0], 2.0),
            Neuron::new([0.5, -0.91, 0.26, -0.5], 3.0),
            Neuron::new([-0.26, -0.27, 0.17, 0.87], 0.5),
        ]);
        assert!(
            LAYER
                .forward_batch(INPUTS)
                .flatten()
                .zip(
                    [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]
                        .into_iter()
                        .flatten()
                )
                .all(|(left, right)| float_equal(left, right))
        )
    }

    #[test]
    fn multiple_layers() {
        const INPUTS: [[f64; 4]; 3] = [
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8],
        ];
        const LAYER: Dense<4, 3> = Dense::new([
            Neuron::new([0.2, 0.8, -0.5, 1.0], 2.0),
            Neuron::new([0.5, -0.91, 0.26, -0.5], 3.0),
            Neuron::new([-0.26, -0.27, 0.17, 0.87], 0.5),
        ]);
        const LAYER2: Dense<3, 3> = Dense::new([
            Neuron::new([0.1, -0.14, 0.5], -1.0),
            Neuron::new([-0.5, 0.12, -0.33], 2.0),
            Neuron::new([-0.44, 0.73, -0.13], -0.5),
        ]);
        assert!(
            LAYER2
                .forward_batch(LAYER.forward_batch(INPUTS))
                .flatten()
                .zip(
                    [
                        [0.5031, -1.04185, -2.03875],
                        [0.2434, -2.7332, -5.7633],
                        [-0.99314, 1.41254, -0.35655]
                    ]
                    .into_iter()
                    .flatten()
                )
                .all(|(left, right)| float_equal(left, right))
        )
    }

    #[test]
    fn layer_with_dataset() {
        let (x, _) = spiral(100, 3);
        let dense = random::<Dense<2, 3>>();
        let mut output = dense.forward_batch(x);
        assert!(output.next().unwrap().into_iter().all(|value| value == 0.0));
        assert!(output.take(5).flatten().all(|value| value != 0.0));
    }
}
