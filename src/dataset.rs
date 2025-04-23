use core::{f64, iter};

pub fn spiral(
    samples: usize,
    classes: usize,
) -> (impl Iterator<Item = [f64; 2]>, impl Iterator<Item = f64>) {
    let x = (0..classes).flat_map(move |class_number| {
        let t_offset = class_number as f64 * 4.0;
        (0..samples)
            .zip(rand::random_iter::<f64>())
            .map(move |(sample_number, noise)| {
                let r = 1.0 / samples as f64 * sample_number as f64;
                let t = t_offset + 4.0 / samples as f64 * sample_number as f64 + noise;
                let (x, y) = t.sin_cos();
                [r * x, r * y]
            })
    });
    let y = (0..classes).flat_map(move |class_number| {
        iter::repeat_n(class_number, samples).map(|label| label as f64)
    });
    (x, y)
}

pub fn sine(samples: usize) -> (impl Iterator<Item = [f64; 1]>, impl Iterator<Item = f64>) {
    let x = (0..samples).map(move |sample| [sample as f64 / samples as f64]);
    let y = (0..samples)
        .map(move |sample| (2.0 * f64::consts::PI * (sample as f64 / samples as f64)).sin());
    (x, y)
}

pub fn vertical(
    samples: usize,
    classes: usize,
) -> (impl Iterator<Item = [f64; 2]>, impl Iterator<Item = f64>) {
    let x = (0..classes).flat_map(move |class_number| {
        (0..samples)
            .zip(rand::random_iter::<f64>())
            .map(move |(_, noise)| [noise * 0.1 + class_number as f64 / 3.0, noise * 0.1 + 0.5])
    });
    let y = (0..classes).flat_map(move |class_number| {
        iter::repeat_n(class_number, samples).map(|sample| sample as f64)
    });
    (x, y)
}
