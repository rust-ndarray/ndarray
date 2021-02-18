#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::float_cmp
)]

use ndarray::Array;
use ndarray::IntoDimension;
use ndarray::ShapeBuilder;
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
fn test_ixdyn_iterate() {
    for &rev in &[false, true] {
        let mut a = Array::zeros((2, 3, 4).set_f(rev));
        let dim = a.shape().to_vec();
        for (i, elt) in a.iter_mut().enumerate() {
            *elt = i;
        }
        println!("{:?}", a.dim());
        let mut a = a.into_shape(dim).unwrap();
        println!("{:?}", a.dim());
        let mut c = 0;
        for (i, elt) in a.iter_mut().enumerate() {
            assert_eq!(i, *elt);
            c += 1;
        }
        assert_eq!(c, a.len());
    }
}

#[test]
fn test_ixdyn_index_iterate() {
    for &rev in &[false, true] {
        let mut a = Array::zeros((2, 3, 4).set_f(rev));
        let dim = a.shape().to_vec();
        for ((i, j, k), elt) in a.indexed_iter_mut() {
            *elt = i + 10 * j + 100 * k;
        }
        let a = a.into_shape(dim).unwrap();
        println!("{:?}", a.dim());
        let mut c = 0;
        for (i, elt) in a.indexed_iter() {
            assert_eq!(a[i], *elt);
            c += 1;
        }
        assert_eq!(c, a.len());
    }
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

#[test]
fn test_0() {
    let mut a = Array::zeros(vec![]);
    let z = vec![].into_dimension();
    assert_eq!(a[z.clone()], 0.);
    a[[]] = 1.;
    assert_eq!(a[[]], 1.);
    assert_eq!(a.len(), 1);
    assert!(!a.is_empty());
    assert_eq!(a.as_slice().unwrap(), &[1.]);

    let mut a = Array::zeros(vec![].f());
    assert_eq!(a[[]], 0.);
    a[[]] = 1.;
    assert_eq!(a[[]], 1.);
    assert_eq!(a.len(), 1);
    assert!(!a.is_empty());
    assert_eq!(a.as_slice().unwrap(), &[1.]);
}

#[test]
fn test_0_add() {
    let mut a = Array::zeros(vec![]);
    a += 1.;
    assert_eq!(a[[]], 1.);
    a += 2.;
    assert_eq!(a[[]], 3.);
}

#[test]
fn test_0_add_add() {
    let mut a = Array::zeros(vec![]);
    a += 1.;
    let mut b = Array::zeros(vec![]);
    b += 1.;
    a += &b;
    assert_eq!(a[[]], 2.);
}

#[test]
fn test_0_add_broad() {
    let mut b = Array::from(vec![5., 6.]);
    let mut a = Array::zeros(vec![]);
    a += 1.;
    b += &a;
    assert_eq!(b[0], 6.);
    assert_eq!(b[1], 7.);
}

#[test]
#[cfg(feature = "std")]
fn test_into_dimension() {
    use ndarray::{Ix0, Ix1, Ix2, IxDyn};

    let a = Array::linspace(0., 41., 6 * 7).into_shape((6, 7)).unwrap();
    let a2 = a.clone().into_shape(IxDyn(&[6, 7])).unwrap();
    let b = a2.clone().into_dimensionality::<Ix2>().unwrap();
    assert_eq!(a, b);

    assert!(a2.clone().into_dimensionality::<Ix0>().is_err());
    assert!(a2.clone().into_dimensionality::<Ix1>().is_err());
    assert!(a2.clone().into_dimensionality::<Ix3>().is_err());

    let c = a2.clone().into_dimensionality::<IxDyn>().unwrap();
    assert_eq!(a2, c);
}
