
extern crate ndarray;
extern crate ndarray_parallel;

use ndarray::prelude::*;
use ndarray_parallel::prelude::*;

use ndarray::Zip;

const M: usize = 1024 * 10;
const N: usize = 100;

#[test]
fn test_zip_1() {
    let mut a = Array2::<f64>::zeros((M, N));

    Zip::from(&mut a)
        .par_apply(|x| {
            *x = x.exp()
        });
}

#[test]
fn test_zip_index_1() {
    let mut a = Array2::default((10, 10));

    Zip::indexed(&mut a)
        .par_apply(|i, x| {
            *x = i;
        });

    for (i, elt) in a.indexed_iter() {
        assert_eq!(*elt, i);
    }
}

#[test]
fn test_zip_index_2() {
    let mut a = Array2::default((M, N));

    Zip::indexed(&mut a)
        .par_apply(|i, x| {
            *x = i;
        });

    for (i, elt) in a.indexed_iter() {
        assert_eq!(*elt, i);
    }
}

#[test]
fn test_zip_index_3() {
    let mut a = Array::default((1, 2, 1, 2, 3));

    Zip::indexed(&mut a)
        .par_apply(|i, x| {
            *x = i;
        });

    for (i, elt) in a.indexed_iter() {
        assert_eq!(*elt, i);
    }
}

#[test]
fn test_zip_index_4() {
    let mut a = Array2::zeros((M, N));
    let mut b = Array2::zeros((M, N));

    Zip::indexed(&mut a)
        .and(&mut b)
        .par_apply(|(i, j), x, y| {
            *x = i;
            *y = j;
        });

    for ((i, _), elt) in a.indexed_iter() {
        assert_eq!(*elt, i);
    }
    for ((_, j), elt) in b.indexed_iter() {
        assert_eq!(*elt, j);
    }
}
