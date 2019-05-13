extern crate approx;
extern crate itertools;
extern crate ndarray;
extern crate ndarray_parallel;

use approx::assert_abs_diff_eq;
use itertools::enumerate;
use ndarray::prelude::*;
use ndarray_parallel::par_azip;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn test_par_azip1() {
    let mut a = Array::zeros(62);
    let b = Array::from_elem(62, 42);
    par_azip!(mut a in { *a = 42 });
    assert_eq!(a, b);
}

#[test]
fn test_par_azip2() {
    let mut a = Array::zeros((5, 7));
    let b = Array::from_shape_fn(a.dim(), |(i, j)| 1. / (i + 2 * j) as f32);
    par_azip!(mut a, b in { *a = b; });
    assert_eq!(a, b);
}

#[test]
fn test_par_azip3() {
    let mut a = [0.; 32];
    let mut b = [0.; 32];
    let mut c = [0.; 32];
    for (i, elt) in enumerate(&mut b) {
        *elt = i as f32;
    }

    par_azip!(mut a (&mut a[..]), b (&b[..]), mut c (&mut c[..]) in {
        *a += b / 10.;
        *c = a.sin();
    });
    let res = Array::linspace(0., 3.1, 32).mapv_into(f32::sin);
    assert_abs_diff_eq!(res, ArrayView::from(&c), epsilon = 1e-4);
}

#[should_panic]
#[test]
fn test_zip_dim_mismatch_1() {
    let mut a = Array::zeros((5, 7));
    let mut d = a.raw_dim();
    d[0] += 1;
    let b = Array::from_shape_fn(d, |(i, j)| 1. / (i + 2 * j) as f32);
    par_azip!(mut a, b in { *a = b; });
}

#[test]
fn test_indices_1() {
    let mut a1 = Array::default(12);
    for (i, elt) in a1.indexed_iter_mut() {
        *elt = i;
    }

    let count = AtomicUsize::new(0);
    par_azip!(index i, elt (&a1) in {
        count.fetch_add(1, Ordering::SeqCst);
        assert_eq!(elt, i);
    });
    assert_eq!(count.load(Ordering::SeqCst), a1.len());
}
