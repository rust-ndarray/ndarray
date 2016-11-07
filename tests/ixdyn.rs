
extern crate ndarray;

use ndarray::Array;
use ndarray::Ix3;

#[test]
fn test_ixdyn() {
    // check that we can use fixed size arrays for indexing
    let mut a = Array::zeros(vec![2, 3, 4]);
    a[[1, 1, 1]] = 1.;
    assert_eq!(a[[1, 1, 1]], 1.);
}

#[should_panic]
#[test]
fn test_ixdyn_wrong_dim() {
    // check that we can use but it panics at runtime, if number of axes is wrong
    let mut a = Array::zeros(vec![2, 3, 4]);
    a[[1, 1, 1]] = 1.;

    let _ = a[[0, 0]];
}

#[test]
fn test_ixdyn_out_of_bounds() {
    // check that we are out of bounds
    let a = Array::<f32, _>::zeros(vec![2, 3, 4]);
    let res = a.get([0, 3, 0]);
    assert_eq!(res, None);
}

#[test]
fn test_ixdyn_uget() {
    // check that we are out of bounds
    let mut a = Array::<f32, _>::zeros(vec![2, 3, 4]);

    a[[1, 2, 0]] = 1.;
    a[[1, 2, 1]] = 2.;
    a[[1, 2, 3]] = 7.;

    let mut x = Ix3(1, 2, 0);
    let step = Ix3(0, 0, 1);
    let mut sum = 0.;
    while let Some(&v) = a.get(x) {
        sum += v;
        x += step;
    }
    assert_eq!(sum, 10.);

    let mut x = Ix3(1, 2, 0);
    let mut sum = 0.;
    unsafe {
        for _ in 0..4 {
            sum += *a.uget(x);
            x += step;
        }
    }
    assert_eq!(sum, 10.);
}
