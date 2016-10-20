#![feature(test)]

extern crate test;
use test::Bencher;

#[macro_use(s)]
extern crate ndarray;
use ndarray::prelude::*;

#[bench]
fn default_f64(bench: &mut Bencher) {
    bench.iter(|| {
        Array::<f64, _>::default((128, 128))
    })
}

#[bench]
fn zeros_f64(bench: &mut Bencher) {
    bench.iter(|| {
        Array::<f64, _>::zeros((128, 128))
    })
}
