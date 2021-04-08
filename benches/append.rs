#![feature(test)]

extern crate test;
use test::Bencher;

use ndarray::prelude::*;

#[bench]
fn select_axis0(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((256, 256));
    let selectable = vec![0, 1, 2, 0, 1, 3, 0, 4, 16, 32, 128, 147, 149, 220, 221, 255, 221, 0, 1];
    bench.iter(|| {
        a.select(Axis(0), &selectable)
    });
}

#[bench]
fn select_axis1(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros((256, 256));
    let selectable = vec![0, 1, 2, 0, 1, 3, 0, 4, 16, 32, 128, 147, 149, 220, 221, 255, 221, 0, 1];
    bench.iter(|| {
        a.select(Axis(1), &selectable)
    });
}

#[bench]
fn select_1d(bench: &mut Bencher) {
    let a = Array::<f32, _>::zeros(1024);
    let mut selectable = (0..a.len()).step_by(17).collect::<Vec<_>>();
    selectable.extend(selectable.clone().iter().rev());

    bench.iter(|| {
        a.select(Axis(0), &selectable)
    });
}
