#![feature(test)]

extern crate test;
use test::Bencher;

use ndarray::prelude::*;

const N: usize = 1024;
const X: usize = 64;
const Y: usize = 16;

#[bench]
fn clip(bench: &mut Bencher) {
    let mut a = Array::linspace(0., 127., N * 2)
        .into_shape([X, Y * 2])
        .unwrap();
    let min = 2.;
    let max = 5.;
    bench.iter(|| {
        a.mapv_inplace(|mut x| {
            if x < min {
                x = min
            }
            if x > max {
                x = max
            }
            x
        })
    });
}
