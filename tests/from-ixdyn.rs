extern crate ndarray;

use ndarray::prelude::*;
use ndarray::{ErrorKind, ShapeError};

#[test]
fn test_arr0_from_dyn() {
    let arr = Array0::from_elem((), 1i32);
    let brr = Array0::from_dyn(arr.clone().into_dyn()).unwrap();

    assert_eq!(arr, brr);
}

#[test]
fn test_arr0_from_dyn_nonstandard_strides() {
    let arr = Array0::from_elem((), 1i32);
    let brr = Array0::from_dyn(arr.clone().into_dyn()).unwrap();

    assert_eq!(arr, brr);
}

#[test]
fn test_arr2_from_dyn() {
    let arr = Array2::from_shape_fn((12, 34), |(i, j)| i * 34 + j);
    let brr = Array2::from_dyn(arr.clone().into_dyn()).unwrap();

    assert_eq!(arr, brr);
}

#[test]
fn test_arr2_from_dyn_nonstandard_strides() {
    let arr = Array2::from_shape_fn((12, 34).f(), |(i, j)| i * 34 + j);
    let brr = Array2::from_dyn(arr.clone().into_dyn()).unwrap();

    assert_eq!(arr, brr);
}

#[test]
fn test_aview2_from_dyn() {
    let arr = Array2::from_shape_fn((12, 34), |(i, j)| i * 34 + j);
    let aview = arr.view();
    let bview = ArrayView2::from_dyn(aview.into_dyn()).unwrap();

    assert_eq!(aview, bview);
}

#[test]
fn test_aview2_from_dyn_nonstandard_strides() {
    let arr = Array2::from_shape_fn((12, 34).f(), |(i, j)| i * 34 + j);
    let aview = arr.view();
    let bview = ArrayView2::from_dyn(aview.into_dyn()).unwrap();

    assert_eq!(aview, bview);
}

#[test]
fn test_arr3_from_incompatible_dyn() {
    let arr = Array2::from_shape_fn((12, 34), |(i, j)| i * 34 + j);
    let brr = Array3::from_dyn(arr.clone().into_dyn());

    assert_eq!(
        brr,
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape))
    );
}

#[test]
fn test_arrd_from_dyn() {
    let arr = Array2::from_shape_fn((12, 34), |(i, j)| i * 34 + j).into_dyn();
    let brr = ArrayD::from_dyn(arr.clone());

    assert_eq!(
        brr,
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape))
    );
}
