#![cfg(feature = "rayon")]

extern crate rayon;
extern crate ndarray;

use ndarray::prelude::*;

use rayon::par_iter::ParallelIterator;

#[test]
fn test_axis_iter() {
    let mut a = Array2::<u32>::zeros((10240, 10240));
    for (i, mut v) in a.axis_iter_mut(Axis(0)).enumerate() {
        v.fill(i as _);
    }
    let s = ParallelIterator::map(a.axis_iter(Axis(0)), |x| x.scalar_sum()).sum();
    assert_eq!(s, a.scalar_sum());
}
