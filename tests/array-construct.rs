
extern crate ndarray;

use ndarray::prelude::*;

#[test]
fn test_from_shape_fn() {
    let step = 3.1;
    let h = Array::from_shape_fn((5, 5),
        |(i, j)| f64::sin(i as f64 / step) * f64::cos(j as f64  / step));
    assert_eq!(h.shape(), &[5, 5]);
}
