pub fn spiral(samples: u8, classes: u8) -> (Vec<[f64; 2]>, Vec<u8>) {
    let mut data = Vec::with_capacity(usize::from(samples) * usize::from(classes));
    let mut labels = Vec::with_capacity(usize::from(samples) * usize::from(classes));

    for class_number in 0..classes {
        let r = (0..samples)
            .map(|sample| 1.0 / f64::from(samples - 1) * f64::from(sample))
            .collect::<Vec<_>>();
        let t = (0..samples)
            .map(|sample| {
                f64::from(class_number)
                    .mul_add(4.0, 4.0 / (f64::from(samples) - 1.0) * f64::from(sample))
            })
            .collect::<Vec<_>>();
        for (r, t) in r.into_iter().zip(t) {
            data.push([r * (t * 2.5).sin(), r * (t * 2.5).cos()]);
        }
        labels.extend((0..samples).map(|_| class_number));
    }

    (data, labels)
}
