use neural_networks_from_scratch::{activation::relu::ReLu, dataset::spiral, layer::Dense};
use rand::random;

fn main() {
    let (x, _) = spiral(100, 3);
    let dense = random::<Dense<2, 3>>();
    let output = ReLu.forward_batch(dense.forward_batch(x));
    println!("{:?}", &output.collect::<Vec<_>>()[..5]);
}
