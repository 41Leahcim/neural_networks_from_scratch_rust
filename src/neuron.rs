pub struct Neuron<const SIZE: usize> {
    weights: [f64; SIZE],
    bias: f64,
}

impl<const SIZE: usize> Neuron<SIZE> {
    pub const fn new(weights: [f64; SIZE], bias: f64) -> Self {
        Self { weights, bias }
    }

    pub const fn forward(&self, inputs: &[f64; SIZE]) -> f64 {
        let mut result = self.bias;
        let mut i = 0;
        while i < inputs.len() {
            result += self.weights[i] * inputs[i];
            i += 1;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::float_equal;

    use super::Neuron;

    #[test]
    fn test1() {
        let inputs = [1.0, 2.0, 3.0];
        let neuron = Neuron::new([0.2, 0.8, -0.5], 2.0);
        let output = neuron.forward(&inputs);
        assert!(float_equal(output, 2.3));
    }

    #[test]
    fn test2() {
        let inputs = [1.0, 2.0, 3.0, 2.5];
        let neuron = Neuron::new([0.2, 0.8, -0.5, 1.0], 2.0);
        let output = neuron.forward(&inputs);
        assert!(float_equal(output, 4.8));
    }
}
