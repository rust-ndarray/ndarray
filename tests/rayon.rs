#![cfg(feature = "rayon")]

extern crate rayon;
#[macro_use(s)] extern crate ndarray;

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

#[test]
fn test_axis_iter_mut() {
    let mut a = Array2::<u32>::zeros((1024 * 1024, 100));
    a.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut v)| v.fill(i as _));
    assert_eq!(a.scalar_sum(),
               (0..a.len_of(Axis(0))).map(|n| n as u32 * a.len_of(Axis(1)) as u32).sum::<u32>());
    println!("{:?}", a.slice(s![..10, ..10]));
}
