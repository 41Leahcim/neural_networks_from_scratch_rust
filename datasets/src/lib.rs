#![warn(clippy::pedantic, clippy::nursery)]

use std::f64::consts::PI;

use math::linspace;

/// Creates a sinewave dataset of a specified size
///
/// # Arguments
/// ```samples```: the number of samples
///
/// # Returns
/// A matrix containing samples, and a matrix containing the labels
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn sine(samples: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let x = (0..samples)
        .map(|i| vec![i as f64 / samples as f64])
        .collect::<Vec<Vec<f64>>>();
    let y = x
        .iter()
        .map(|array| {
            array
                .iter()
                .map(|&value| (value * 2.0 * PI).sin())
                .collect()
        })
        .collect();

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
pub fn vertical(samples: usize, classes: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // create the outer samples vector
    let mut x = Vec::with_capacity(samples * classes);

    // create the outer labels vector
    let mut y = Vec::with_capacity(samples * classes);

    // create the data, per class first
    (0..classes).for_each(|class| {
        // create the samples and set the current class of the current line
        (0..samples).for_each(|_| {
            // Generate the data
            x.push(vec![
                (rand::random::<f64>() % (samples as f64)).mul_add(0.1, class as f64 / 3.0),
                (rand::random::<f64>() % (samples as f64)).mul_add(0.1, 0.5),
            ]);

            // Set the label
            y.push(vec![class as f64]);
        });
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
pub fn spiral(samples: usize, classes: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // create the outer samples vector
    let mut x = Vec::with_capacity(samples * classes);

    // create the outer labels vector
    let mut y = Vec::with_capacity(samples * classes);

    // create the data
    (0..classes).for_each(|class| {
        // create two arrays of linear data
        let r = linspace(0.0, 1.0, samples);
        let mut t = linspace(class as f64 * 4.0, (class as f64 + 1.0) * 4.0, samples);

        // generate the data for the current class
        (0..samples).for_each(|sample| {
            // add a random value to the current t-value
            t[sample] += rand::random::<f64>() * 0.2;

            // generate the samples
            x.push(vec![
                r[sample] * (t[sample] * 2.5).sin(),
                r[sample] * (t[sample] * 2.5).cos(),
            ]);

            // set the label
            y.push(vec![class as f64]);
        });
    });

    // return the result
    (x, y)
}
