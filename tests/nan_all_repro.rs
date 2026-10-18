// Regression test for https://github.com/rust-ndarray/ndarray/issues/1612
//
// `is_all_nan`, `is_any_nan`, `is_all_infinite`, and `is_any_infinite` were
// inverted: the `$all` predicate tested `!elt.$func()` ("all NOT nan") and the
// `$any` predicate rode on the double negation, so `is_all_nan` returned `true`
// for an all-finite array. These tests pin the correct behavior for both the
// NaN and infinite families.
#![cfg(feature = "std")]

use ndarray::array;

#[test]
fn is_all_nan_and_is_any_nan()
{
    // All finite: neither all nor any element is NaN.
    let all_finite = array![1.0_f64, 2.0, 3.0];
    assert!(!all_finite.is_all_nan());
    assert!(!all_finite.is_any_nan());

    // All NaN: every element is NaN.
    let all_nan = array![f64::NAN, f64::NAN, f64::NAN];
    assert!(all_nan.is_all_nan());
    assert!(all_nan.is_any_nan());

    // Mixed: not all elements are NaN, but at least one is.
    let mixed = array![1.0_f64, f64::NAN, 3.0];
    assert!(!mixed.is_all_nan());
    assert!(mixed.is_any_nan());
}

#[test]
fn is_all_infinite_and_is_any_infinite()
{
    // All finite: neither all nor any element is infinite.
    let all_finite = array![1.0_f64, 2.0, 3.0];
    assert!(!all_finite.is_all_infinite());
    assert!(!all_finite.is_any_infinite());

    // All infinite: every element is infinite.
    let all_inf = array![f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY];
    assert!(all_inf.is_all_infinite());
    assert!(all_inf.is_any_infinite());

    // Mixed: not all elements are infinite, but at least one is.
    let mixed = array![1.0_f64, f64::INFINITY, 3.0];
    assert!(!mixed.is_all_infinite());
    assert!(mixed.is_any_infinite());
}
