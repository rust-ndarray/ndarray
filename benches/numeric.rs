
#![feature(test)]
extern crate test;
use test::{black_box, Bencher};

extern crate ndarray;
use ndarray::prelude::*;

const N: usize = 1024;
const X: usize = 64;
const Y: usize = 16;

#[bench]
fn clip(bench: &mut Bencher)
{
    let mut a = Array::linspace(0., 127., N * 2).into_shape([X, Y * 2]).unwrap();
    let min = 2.;
    let max = 5.;
    bench.iter(|| {
        a.mapv_inplace(|mut x| {
            if x < min { x = min }
            if x > max { x = max }
            x
        })
    });
}

#[bench]
fn contiguous_sum_int_1e4(bench: &mut Bencher)
{
    let n = 1e4 as usize;
    let a = Array::from_vec((0..n).collect());
    bench.iter(|| {
        a.sum()
    });
}

#[bench]
fn contiguous_sum_1e7(bench: &mut Bencher)
{
    let n = 1e7 as usize;
    let a = Array::linspace(-1e6, 1e6, n);
    bench.iter(|| {
        a.sum()
    });
}

#[bench]
fn contiguous_sum_1e4(bench: &mut Bencher)
{
    let n = 1e4 as usize;
    let a = Array::linspace(-1e6, 1e6, n);
    bench.iter(|| {
        a.sum()
    });
}

#[bench]
fn contiguous_sum_1e2(bench: &mut Bencher)
{
    let n = 1e2 as usize;
    let a = Array::linspace(-1e6, 1e6, n);
    bench.iter(|| {
        a.sum()
    });
}

#[bench]
fn contiguous_sum_ix3_1e2(bench: &mut Bencher)
{
    let n = 1e2 as usize;
    let a = Array::linspace(-1e6, 1e6, n * n * n)
        .into_shape([n, n, n])
        .unwrap();
    bench.iter(|| black_box(&a).sum());
}

#[bench]
fn inner_discontiguous_sum_ix3_1e2(bench: &mut Bencher)
{
    let n = 1e2 as usize;
    let a = Array::linspace(-1e6, 1e6, n * n * 2*n)
        .into_shape([n, n, 2*n])
        .unwrap();
    let v = a.slice(s![.., .., ..;2]);
    bench.iter(|| black_box(&v).sum());
}

#[bench]
fn middle_discontiguous_sum_ix3_1e2(bench: &mut Bencher)
{
    let n = 1e2 as usize;
    let a = Array::linspace(-1e6, 1e6, n * 2*n * n)
        .into_shape([n, 2*n, n])
        .unwrap();
    let v = a.slice(s![.., ..;2, ..]);
    bench.iter(|| black_box(&v).sum());
}

#[bench]
fn sum_by_row_1e4(bench: &mut Bencher)
{
    let n = 1e3 as usize;
    let a = Array::linspace(-1e6, 1e6, n * n)
        .into_shape([n, n])
        .unwrap();
    bench.iter(|| {
        a.sum_axis(Axis(0))
    });
}

#[bench]
fn sum_by_col_1e4(bench: &mut Bencher)
{
    let n = 1e3 as usize;
    let a = Array::linspace(-1e6, 1e6, n * n)
        .into_shape([n, n])
        .unwrap();
    bench.iter(|| {
        a.sum_axis(Axis(1))
    });
}

#[bench]
fn sum_by_middle_1e2(bench: &mut Bencher)
{
    let n = 1e2 as usize;
    let a = Array::linspace(-1e6, 1e6, n * n * n)
        .into_shape([n, n, n])
        .unwrap();
    bench.iter(|| {
        a.sum_axis(Axis(1))
    });
}
