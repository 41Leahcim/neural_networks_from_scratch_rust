use nnfs::{dataset, DenseLayer};

fn main() {
    let data = dataset::spiral(100, 3);
    let layer = DenseLayer::<2, 3>::random();
    let output = layer.forward_batch(&data.0);
    println!("{:?}", &output[..5]);
    assert_eq!(output.len(), 300);
    assert_eq!(output[0].len(), 3);
}
