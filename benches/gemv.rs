#![cfg(feature = "nightly")]
#![feature(test)]
#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names
)]

extern crate test;
use test::Bencher;

use ndarray::prelude::*;

use ndarray::linalg::general_mat_vec_mul;

#[bench]
fn gemv_64_64c(bench: &mut Bencher) {
    let a = Array::zeros((64, 64));
    let (m, n) = a.dim();
    let x = Array::zeros(n);
    let mut y = Array::zeros(m);
    bench.iter(|| {
        general_mat_vec_mul(1.0, &a, &x, 1.0, &mut y);
    });
}

#[bench]
fn gemv_64_64f(bench: &mut Bencher) {
    let a = Array::zeros((64, 64).f());
    let (m, n) = a.dim();
    let x = Array::zeros(n);
    let mut y = Array::zeros(m);
    bench.iter(|| {
        general_mat_vec_mul(1.0, &a, &x, 1.0, &mut y);
    });
}

#[bench]
fn gemv_64_32(bench: &mut Bencher) {
    let a = Array::zeros((64, 32));
    let (m, n) = a.dim();
    let x = Array::zeros(n);
    let mut y = Array::zeros(m);
    bench.iter(|| {
        general_mat_vec_mul(1.0, &a, &x, 1.0, &mut y);
    });
}
