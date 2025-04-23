use crate::neuron::Neuron;

pub struct Layer<const INPUT: usize, const OUTPUT: usize> {
    neurons: [Neuron<INPUT>; OUTPUT],
}

impl<const INPUT: usize, const OUTPUT: usize> Layer<INPUT, OUTPUT> {
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

#[cfg(test)]
mod tests {
    use crate::{float_equal, neuron::Neuron};

    use super::Layer;

    #[test]
    fn layer_with_sample() {
        const INPUTS: [f64; 4] = [1.0, 2.0, 3.0, 2.5];
        const LAYER: Layer<4, 3> = Layer::new([
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
        const LAYER: Layer<4, 3> = Layer::new([
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
}
