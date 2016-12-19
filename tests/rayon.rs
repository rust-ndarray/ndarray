#![cfg(feature = "rayon")]

extern crate rayon;
extern crate ndarray;

use ndarray::prelude::*;

use rayon::prelude::*;

#[test]
fn test_axis_iter() {
    let mut a = Array2::<u32>::zeros((1024 * 1024, 100));
    for (i, mut v) in a.axis_iter_mut(Axis(0)).enumerate() {
        v.fill(i as _);
    }
    assert_eq!(a.axis_iter(Axis(0)).len(), 1024 * 1024);
    let s = a.axis_iter(Axis(0)).into_par_iter().map(|x| x.scalar_sum()).sum();
    assert_eq!(s, a.scalar_sum());
}
