#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::float_cmp
)]

use ndarray::Array;
use ndarray::Ix0;
use ndarray::ShapeBuilder;

#[test]
fn test_ix0() {
    let mut a = Array::zeros(Ix0());
    assert_eq!(a[()], 0.);
    a[()] = 1.;
    assert_eq!(a[()], 1.);
    assert_eq!(a.len(), 1);
    assert!(!a.is_empty());
    assert_eq!(a.as_slice().unwrap(), &[1.]);

    let mut a = Array::zeros(Ix0().f());
    assert_eq!(a[()], 0.);
    a[()] = 1.;
    assert_eq!(a[()], 1.);
    assert_eq!(a.len(), 1);
    assert!(!a.is_empty());
    assert_eq!(a.as_slice().unwrap(), &[1.]);
}

#[test]
fn test_ix0_add() {
    let mut a = Array::zeros(Ix0());
    a += 1.;
    assert_eq!(a[()], 1.);
    a += 2.;
    assert_eq!(a[()], 3.);
}

#[test]
fn test_ix0_add_add() {
    let mut a = Array::zeros(Ix0());
    a += 1.;
    let mut b = Array::zeros(Ix0());
    b += 1.;
    a += &b;
    assert_eq!(a[()], 2.);
}

#[test]
fn test_ix0_add_broad() {
    let mut b = Array::from(vec![5., 6.]);
    let mut a = Array::zeros(Ix0());
    a += 1.;
    b += &a;
    assert_eq!(b[0], 6.);
    assert_eq!(b[1], 7.);
}
