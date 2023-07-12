#![warn(clippy::pedantic, clippy::nursery)]

use std::ops::{Add, Mul};

use ndarray::{s, Array, Array2};
use rand::Rng;

/// Creates a sinewave dataset of a specified size
///
/// # Arguments
/// ```samples```: the number of samples
///
/// # Returns
/// A matrix containing samples, and a matrix containing the labels
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn sine(samples: usize) -> (Array2<f64>, Array2<f64>) {
    let x = Array::linspace(0.0, 1.0, samples).insert_axis(ndarray::Axis(1));
    let y = Array::from_shape_fn((1, samples), |(_, j)| {
        (2.0 * std::f64::consts::PI * x[[j, 0]]).sin()
    })
    .reversed_axes();

    (x, y)
}

/// Creates a vertical lines dataset of a specified size (samples), containing a specified number of lines (classes)
///
/// # Arguments
/// ```samples```: the number of points per line
/// ```classes```: the number of lines
///
/// # Returns
/// A matrix containing samples, and a matrix containing labels
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn vertical(samples: usize, classes: usize) -> (Array2<f64>, Array2<f64>) {
    let mut x = Array2::<f64>::zeros((samples * classes, 2));
    let mut y = Array2::<f64>::zeros((samples * classes, 1));

    (0..classes).for_each(|class_number| {
        let ix = samples * class_number..samples * class_number.add(1);
        x.slice_mut(s![ix.clone(), ..])
            .assign(&Array::from_shape_fn((samples, 2), |(_, j)| {
                let class_f = class_number as f64;
                let mut rng = rand::thread_rng();
                match j {
                    0 => rng.gen::<f64>().mul_add(0.1, class_f / 3.0),
                    1 => rng.gen::<f64>().mul_add(0.1, 0.5),
                    _ => unreachable!(),
                }
            }));
        y.slice_mut(s![ix, ..]).fill(class_number as f64);
    });

    // return the data samples and labels
    (x, y)
}

/// Creates one spiral per class, each spiral containing a specified number of samples
///
/// # Arguments
/// ```samples```: the number of points per spiral
/// ```classes```: the number of spirals
///
/// # Returns
/// A matrix containing data for a number of spirals
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn spiral(samples: usize, classes: usize) -> (Array2<f64>, Array2<f64>) {
    let mut x = Array::zeros((samples * classes, 2));
    let mut y = Array::zeros((samples * classes, 1));

    (0..classes).for_each(|class_number| {
        let ix = samples * class_number..samples * (class_number + 1);
        let r = Array::linspace(0.0, 1.0, samples);
        let t = Array::linspace(
            (class_number * 4) as f64,
            class_number.add(1).mul(4) as f64,
            samples,
        ) + Array::<f64, _>::zeros(samples).mapv(|_| rand::random::<f64>() * 0.2);
        let x_coords = r.clone() * &(t.clone() * 2.5).mapv(f64::sin);
        let y_coords = r * &(t * 2.5).mapv(f64::cos);
        x.slice_mut(s![ix.clone(), 0]).assign(&x_coords);
        x.slice_mut(s![ix.clone(), 1]).assign(&y_coords);
        y.slice_mut(s![ix, 0]).fill(class_number as f64);
    });

    // return the result
    (x, y)
}
