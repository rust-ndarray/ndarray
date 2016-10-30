
#![feature(test)]

extern crate test;
use test::Bencher;

#[macro_use(s)]
extern crate ndarray;
use ndarray::prelude::*;
use ndarray::ArrayViewMut2;

const N: usize = 1024;
const X: usize = 64;
const Y: usize = 16;

#[bench]
fn map_regular(bench: &mut Bencher)
{
    let a = Array::linspace(0., 127., N).into_shape((X, Y)).unwrap();
    bench.iter(|| {
        a.map(|&x| 2. * x)
    });
}


pub fn double_array(mut a: ArrayViewMut2<f64>) {
    a *= 2.0;
}

#[bench]
fn map_stride_double_f64(bench: &mut Bencher)
{
    let mut a = Array::linspace(0., 127., N * 2).into_shape((X, Y * 2)).unwrap();
    let mut av = a.slice_mut(s![.., ..;2]);
    bench.iter(|| {
        double_array(av.view_mut());

    });
}

#[bench]
fn map_stride_f64(bench: &mut Bencher)
{
    let a = Array::linspace(0., 127., N * 2).into_shape((X, Y * 2)).unwrap();
    let av = a.slice(s![.., ..;2]);
    bench.iter(|| {
        av.map(|&x| 2. * x)
    });
}

#[bench]
fn map_stride_u32(bench: &mut Bencher)
{
    let a = Array::linspace(0., 127., N * 2).into_shape((X, Y * 2)).unwrap();
    let b = a.mapv(|x| x as u32);
    let av = b.slice(s![.., ..;2]);
    bench.iter(|| {
        av.map(|&x| 2 * x)
    });
}
