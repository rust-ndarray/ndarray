#![feature(phase)]
#![allow(uppercase_variables)]
#![allow(unused_imports)]

extern crate test;
extern crate ndarray;

use ndarray::{Array, S, Si};
use ndarray::{arr0, arr1, arr2};

use test::black_box;

#[bench]
fn bench_std_add(bench: &mut test::Bencher)
{
    let mut a = arr2::<f32>([[1., 2., 2.],
                             [3., 4., 4.],
                             [3., 4., 4.],
                             [3., 4., 4.],
                             [5., 6., 6.]]);
    let b = a.clone();
    bench.iter(|| a.iadd(&b));
}

#[bench]
fn bench_std_iter_1d(bench: &mut test::Bencher)
{
    let a = arr1::<f32>([1., 2., 2.,
                         3., 4., 4.,
                         3., 4., 4.,
                         3., 4., 4.,
                         5., 6., 6.]);
    bench.iter(|| for elt in a.iter() { black_box(elt) })
}

#[bench]
fn bench_std_iter_2d(bench: &mut test::Bencher)
{
    let a = arr2::<f32>([[1., 2., 2.],
                         [3., 4., 4.],
                         [3., 4., 4.],
                         [3., 4., 4.],
                         [5., 6., 6.]]);
    bench.iter(|| for elt in a.iter() { black_box(elt) })
}
