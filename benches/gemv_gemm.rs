#![feature(test)]
#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names
)]

extern crate test;
use test::Bencher;

use num_complex::Complex;
use num_traits::{Float, One, Zero};

use ndarray::prelude::*;

use ndarray::LinalgScalar;
use ndarray::linalg::general_mat_mul;
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

#[bench]
fn cgemm_100(bench: &mut Bencher) {
    cgemm_bench::<f32>(100, bench);
}

#[bench]
fn zgemm_100(bench: &mut Bencher) {
    cgemm_bench::<f64>(100, bench);
}

fn cgemm_bench<A>(size: usize, bench: &mut Bencher)
where
    A: LinalgScalar + Float,
{
    let (m, k, n) = (size, size, size);
    let a = Array::<Complex<A>, _>::zeros((m, k));

    let x = Array::zeros((k, n));
    let mut y = Array::zeros((m, n));
    bench.iter(|| {
        general_mat_mul(Complex::one(), &a, &x, Complex::zero(), &mut y);
    });
}
