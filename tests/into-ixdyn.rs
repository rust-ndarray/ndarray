extern crate ndarray;

use ndarray::prelude::*;

#[test]
fn test_arr0_into_arrd() {
    let x = 1.23456;
    let mut arr = Array::zeros(Ix0());
    arr[()] = x;
    let brr: ArrayD<_> = arr.into();
    assert!(brr[IxDyn(&[])] == x);
}

#[test]
fn test_arr1_into_arrd() {
    let arr = Array1::from_shape_fn((123), |i| i);
    let brr = arr.clone();
    let crr: ArrayD<_> = brr.into();

    for i in 0..123 {
        assert!(arr[i] == crr[i]);
    }
}

#[test]
fn test_arr2_into_arrd() {
    let arr = Array2::from_shape_fn((12, 34), |(i, j)| i*34 + j);
    let brr = arr.clone();
    let crr: ArrayD<_> = brr.into();

    for i in 0..12 {
        for j in 0..34 {
            assert!(arr[[i, j]] == crr[[i, j]]);
        }
    }
}

#[test]
fn test_arr2_into_arrd_nonstandard_strides() {
    let arr = Array2::from_shape_fn((12, 34).f(), |(i, j)| i*34 + j);
    let brr = arr.clone();
    let crr: ArrayD<_> = brr.into();

    for i in 0..12 {
        for j in 0..34 {
            assert!(arr[[i, j]] == crr[[i, j]]);
        }
    }
}
