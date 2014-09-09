#![feature(phase)]
#![allow(uppercase_variables)]
#![allow(unused_imports)]

extern crate test;
extern crate ndarray;

use ndarray::{Array, S, Si};
use ndarray::{arr0, arr1, arr2};
use ndarray::{d1, d2};

use test::black_box;

#[bench]
fn bench_std_add(bench: &mut test::Bencher)
{
    let mut a = arr2::<f32>(&[&[1., 2., 2.],
                              &[3., 4., 4.],
                              &[3., 4., 4.],
                              &[3., 4., 4.],
                              &[5., 6., 6.]]);
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
    bench.iter(|| for &elt in a.iter() { black_box(elt) })
}

#[bench]
fn bench_std_iter_1d_raw(bench: &mut test::Bencher)
{
    let a = arr1::<f32>([1., 2., 2.,
                         3., 4., 4.,
                         3., 4., 4.,
                         3., 4., 4.,
                         5., 6., 6.]);
    bench.iter(|| for &elt in a.raw_data().iter() { black_box(elt) })
}

#[bench]
fn bench_std_iter_2d(bench: &mut test::Bencher)
{
    let a = arr2::<f32>(&[&[1., 2., 2.],
                          &[3., 4., 4.],
                          &[3., 4., 4.],
                          &[3., 4., 4.],
                          &[5., 6., 6.]]);
    bench.iter(|| for &elt in a.iter() { black_box(elt) })
}

#[bench]
fn bench_std_iter_1d_large(bench: &mut test::Bencher)
{
    let a = Array::<f32, _>::zeros(d1(1024));
    bench.iter(|| for &elt in a.iter() { black_box(elt) })
}

#[bench]
fn bench_std_iter_1d_raw_large(bench: &mut test::Bencher)
{
    let a = Array::<f32, _>::zeros(d1(1024));
    bench.iter(|| for &elt in a.raw_data().iter() { black_box(elt) })
}

#[bench]
fn bench_std_iter_2d_large(bench: &mut test::Bencher)
{
    let a = Array::<f32, _>::zeros(d2(16, 64));
    bench.iter(|| for &elt in a.iter() { black_box(elt) })
}

#[bench]
fn assign_scalar_2d(bench: &mut test::Bencher)
{
    let mut a = arr2::<f32>(&[&[1., 2., 2.],
                              &[3., 4., 4.],
                              &[3., 4., 4.],
                              &[3., 4., 4.],
                              &[5., 6., 6.]]);
    a.swap_axes(0, 1);
    bench.iter(|| a.assign_scalar(&3.))
}

#[bench]
fn bench_iter_diag(bench: &mut test::Bencher)
{
    let a = Array::<f32, _>::zeros(d2(1024, 1024));
    bench.iter(|| for elt in a.diag_iter() { black_box(elt) })
}

#[bench]
fn bench_row_iter(bench: &mut test::Bencher)
{
    let a = Array::<f32, _>::zeros(d2(1024, 1024));
    let it = a.row_iter(17);
    bench.iter(|| for elt in it.clone() { black_box(elt) })
}

#[bench]
fn bench_col_iter(bench: &mut test::Bencher)
{
    let a = Array::<f32, _>::zeros(d2(1024, 1024));
    bench.iter(|| for elt in a.col_iter(17) { black_box(elt) })
}

#[bench]
fn bench_mat_mul(bench: &mut test::Bencher)
{
    let a = arr2::<f32>(&[&[1., 2., 2.],
                          &[3., 4., 4.],
                          &[3., 4., 4.],
                          &[3., 4., 4.],
                          &[5., 6., 6.]]);
    let mut at = a.clone();
    at.swap_axes(0, 1);
    bench.iter(|| at.mat_mul(&a));
}

#[bench]
fn lst_squares(bench: &mut test::Bencher)
{
    let xs = arr2::<f32>(&[&[ 2.,  3.],
                           &[-2., -1.],
                           &[ 1.,  5.],
                           &[-1.,  2.]]);
    let b = arr1([1., -1., 2., 1.]);
    bench.iter(|| ndarray::linalg::least_squares(&xs, &b));
}
