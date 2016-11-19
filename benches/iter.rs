#![feature(test)]

extern crate test;
use test::Bencher;

#[macro_use(s)]
extern crate ndarray;
use ndarray::prelude::*;

#[bench]
fn iter_sum_2d_regular(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((64, 64));
    bench.iter(|| {
        a.iter().fold(0, |acc, &x| acc + x)
    });
}

#[bench]
fn iter_sum_2d_cutout(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = av;
    bench.iter(|| {
        a.iter().fold(0, |acc, &x| acc + x)
    });
}

#[bench]
fn iter_all_2d_cutout(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = av;
    bench.iter(|| {
        a.iter().all(|&x| x >= 0)
    });
}

#[bench]
fn iter_sum_2d_transpose(bench: &mut Bencher)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let a = a.t();
    bench.iter(|| {
        a.iter().fold(0, |acc, &x| acc + x)
    });
}

#[bench]
fn iter_filter_sum_2d_u32(bench: &mut Bencher)
{
    let a = Array::linspace(0., 1., 256).into_shape((16, 16)).unwrap();
    let b = a.mapv(|x| (x * 100.) as u32);
    bench.iter(|| {
        b.iter().filter(|&&x| x < 75).fold(0, |acc, &x| acc + x)
    });
}

#[bench]
fn iter_filter_sum_2d_f32(bench: &mut Bencher)
{
    let a = Array::linspace(0., 1., 256).into_shape((16, 16)).unwrap();
    let b = a * 100.;
    bench.iter(|| {
        b.iter().filter(|&&x| x < 75.).fold(0., |acc, &x| acc + x)
    });
}

#[bench]
fn iter_filter_sum_2d_stride_u32(bench: &mut Bencher)
{
    let a = Array::linspace(0., 1., 256).into_shape((16, 16)).unwrap();
    let b = a.mapv(|x| (x * 100.) as u32);
    let b = b.slice(s![.., ..;2]);
    bench.iter(|| {
        b.iter().filter(|&&x| x < 75).fold(0, |acc, &x| acc + x)
    });
}

#[bench]
fn iter_filter_sum_2d_stride_f32(bench: &mut Bencher)
{
    let a = Array::linspace(0., 1., 256).into_shape((16, 16)).unwrap();
    let b = a * 100.;
    let b = b.slice(s![.., ..;2]);
    bench.iter(|| {
        b.iter().filter(|&&x| x < 75.).fold(0., |acc, &x| acc + x)
    });
}
