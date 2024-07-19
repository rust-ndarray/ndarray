#![allow(
    clippy::many_single_char_names, clippy::deref_addrof, clippy::unreadable_literal, clippy::many_single_char_names,
    clippy::float_cmp
)]

use ndarray::Array;
use ndarray::IxD;
use ndarray::Ix2;

#[test]
fn test_ixd()
{
    // Check that IxD creats a static index based on the provided array size
    let mut a = Array::zeros(IxD([2, 3]));
    assert_eq!(a.raw_dim(), Ix2(2, 3));

    assert_eq!(a[(1, 1)], 0.);

    a[(1, 1)] = 3.;
    assert_eq!(a[(1, 1)], 3.);

    // Wrong index dimension is caught by the type checker
    // a[(1, 1, 1)] = 4.;
}

#[test]
fn test_ixd_repeating()
{
    // Check that repeating creates an array of a specified dimension
    let mut a = Array::zeros(IxD::<2>::repeating(2));
    assert_eq!(a.raw_dim(), Ix2(2, 2));

    a[(1, 1)] = 2.;
    assert_eq!(a[(1, 1)], 2.);
}

