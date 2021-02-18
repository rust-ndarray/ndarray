#![feature(test)]
#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names
)]
extern crate test;
use test::black_box;
use test::Bencher;

use ndarray::prelude::*;

const N: usize = 1024;
const X: usize = 64;
const Y: usize = 16;

#[bench]
fn map_regular(bench: &mut Bencher) {
    let a = Array::linspace(0., 127., N).into_shape((X, Y)).unwrap();
    bench.iter(|| a.map(|&x| 2. * x));
}

pub fn double_array(mut a: ArrayViewMut2<'_, f64>) {
    a *= 2.0;
}

#[bench]
fn map_stride_double_f64(bench: &mut Bencher) {
    let mut a = Array::linspace(0., 127., N * 2)
        .into_shape([X, Y * 2])
        .unwrap();
    let mut av = a.slice_mut(s![.., ..;2]);
    bench.iter(|| {
        double_array(av.view_mut());
    });
}

#[bench]
fn map_stride_f64(bench: &mut Bencher) {
    let a = Array::linspace(0., 127., N * 2)
        .into_shape([X, Y * 2])
        .unwrap();
    let av = a.slice(s![.., ..;2]);
    bench.iter(|| av.map(|&x| 2. * x));
}

#[bench]
fn map_stride_u32(bench: &mut Bencher) {
    let a = Array::linspace(0., 127., N * 2)
        .into_shape([X, Y * 2])
        .unwrap();
    let b = a.mapv(|x| x as u32);
    let av = b.slice(s![.., ..;2]);
    bench.iter(|| av.map(|&x| 2 * x));
}

#[bench]
fn fold_axis(bench: &mut Bencher) {
    let a = Array::linspace(0., 127., N * 2)
        .into_shape([X, Y * 2])
        .unwrap();
    bench.iter(|| a.fold_axis(Axis(0), 0., |&acc, &elt| acc + elt));
}

const MA: usize = 64;
const MASZ: usize = MA * MA;

#[bench]
fn map_axis_0(bench: &mut Bencher) {
    let a = Array::from_iter(0..MASZ as i32)
        .into_shape([MA, MA])
        .unwrap();
    bench.iter(|| a.map_axis(Axis(0), black_box));
}

#[bench]
fn map_axis_1(bench: &mut Bencher) {
    let a = Array::from_iter(0..MASZ as i32)
        .into_shape([MA, MA])
        .unwrap();
    bench.iter(|| a.map_axis(Axis(1), black_box));
}
