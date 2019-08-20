#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::float_cmp
)]

use ndarray::prelude::*;

#[test]
fn test_arr0_into_dyn() {
    assert!(arr0(1.234).into_dyn()[IxDyn(&[])] == 1.234);
}

#[test]
fn test_arr2_into_arrd_nonstandard_strides() {
    let arr = Array2::from_shape_fn((12, 34).f(), |(i, j)| i * 34 + j).into_dyn();
    let brr = ArrayD::from_shape_fn(vec![12, 34], |d| d[0] * 34 + d[1]);

    assert!(arr == brr);
}
