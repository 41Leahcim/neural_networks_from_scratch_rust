use std::time::Instant;

use nnfs::{dataset, DenseLayer};

fn main() {
    let start = Instant::now();
    let data = dataset::spiral(10_000, 1_000);
    let layer = DenseLayer::<2, 3>::random();
    let output = layer.forward_batch(&data.0);
    println!("{:?}", &output[..5]);
    //assert_eq!(output.len(), 300);
    //assert_eq!(output[0].len(), 3);
    println!("{:?}", start.elapsed());
}
