use nnfs::{DenseLayer as Layer, Neuron};

#[test]
fn single_neuron() {
    let inputs = [1.0, 2.0, 3.0];
    let neuron = Neuron::new([0.2, 0.8, -0.5], 2.0);
    let output = neuron.forward(&inputs);
    assert!((output - 2.3).abs() < f64::EPSILON);
    println!("{output}");
}

#[test]
fn neuron_layer() {
    let inputs = [1.0, 2.0, 3.0, 2.5];
    let neurons = Layer::new([
        Neuron::new([0.2, 0.8, -0.5, 1.0], 2.0),
        Neuron::new([0.5, -0.91, 0.26, -0.5], 3.0),
        Neuron::new([-0.26, -0.27, 0.17, 0.87], 0.5),
    ]);
    let output = neurons.forward_sample(&inputs);
    println!("{output:?}");
    assert_eq!(output, [4.8, 1.21, 2.385]);
}

#[test]
fn neuron_layer_with_batch_of_data() {
    let inputs = [
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ];
    let neurons = Layer::new([
        Neuron::new([0.2, 0.8, -0.5, 1.0], 2.0),
        Neuron::new([0.5, -0.91, 0.26, -0.5], 3.0),
        Neuron::new([-0.26, -0.27, 0.17, 0.87], 0.5),
    ]);
    let outputs = neurons.forward_batch(&inputs);
    println!("{outputs:?}");
    assert_eq!(
        outputs,
        [
            [4.8, 1.21, 2.385],
            [8.9, -1.809_999_999_999_999_6, 0.199_999_999_999_999_96],
            [
                1.410_000_000_000_000_1,
                1.050_999_999_999_999_7,
                0.025_999_999_999_999_912
            ]
        ]
    );
}
