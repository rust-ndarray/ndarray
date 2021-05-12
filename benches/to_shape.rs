#![feature(test)]

extern crate test;
use test::Bencher;

use ndarray::prelude::*;
use ndarray::Order;

#[bench]
fn to_shape2_1(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((4, 5));
    let view = a.view();
    bench.iter(|| {
        view.to_shape(4 * 5).unwrap()
    });
}

#[bench]
fn to_shape2_2_same(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((4, 5));
    let view = a.view();
    bench.iter(|| {
        view.to_shape((4, 5)).unwrap()
    });
}

#[bench]
fn to_shape2_2_flip(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((4, 5));
    let view = a.view();
    bench.iter(|| {
        view.to_shape((5, 4)).unwrap()
    });
}

#[bench]
fn to_shape2_3(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((4, 5));
    let view = a.view();
    bench.iter(|| {
        view.to_shape((2, 5, 2)).unwrap()
    });
}

#[bench]
fn to_shape3_1(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((3, 4, 5));
    let view = a.view();
    bench.iter(|| {
        view.to_shape(3 * 4 * 5).unwrap()
    });
}

#[bench]
fn to_shape3_2_order(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((3, 4, 5));
    let view = a.view();
    bench.iter(|| {
        view.to_shape((12, 5)).unwrap()
    });
}

#[bench]
fn to_shape3_2_outoforder(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((3, 4, 5));
    let view = a.view();
    bench.iter(|| {
        view.to_shape((4, 15)).unwrap()
    });
}

#[bench]
fn to_shape3_3c(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((3, 4, 5));
    let view = a.view();
    bench.iter(|| {
        view.to_shape((3, 4, 5)).unwrap()
    });
}

#[bench]
fn to_shape3_3f(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((3, 4, 5).f());
    let view = a.view();
    bench.iter(|| {
        view.to_shape(((3, 4, 5), Order::F)).unwrap()
    });
}

#[bench]
fn to_shape3_4c(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((3, 4, 5));
    let view = a.view();
    bench.iter(|| {
        view.to_shape(((2, 3, 2, 5), Order::C)).unwrap()
    });
}

#[bench]
fn to_shape3_4f(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((3, 4, 5).f());
    let view = a.view();
    bench.iter(|| {
        view.to_shape(((2, 3, 2, 5), Order::F)).unwrap()
    });
}
