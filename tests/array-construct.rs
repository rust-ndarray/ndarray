
extern crate ndarray;

use ndarray::prelude::*;

#[test]
fn test_from_shape_fn() {
    let step = 3.1;
    let h = Array::from_shape_fn((5, 5),
        |(i, j)| f64::sin(i as f64 / step) * f64::cos(j as f64  / step));
    assert_eq!(h.shape(), &[5, 5]);
}

#[test]
fn test_dimension_zero() {
    let a: Array2<f32> = Array2::from(vec![[], [], []]);
    assert_eq!(vec![0.; 0], a.into_raw_vec());
    let a: Array3<f32> = Array3::from(vec![[[]], [[]], [[]]]);
    assert_eq!(vec![0.; 0], a.into_raw_vec());
}

#[test]
fn test_rc_into_owned() {
    let a = Array2::from_elem((5, 5), 1.).into_shared();
    let mut b = a.clone();
    b.fill(0.);
    let mut c = b.into_owned();
    c.fill(2.);
    // test that they are unshared
    assert!(!a.all_close(&c, 0.01));
}
