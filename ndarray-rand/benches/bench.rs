#![feature(test)]

extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate rand_distr;
extern crate test;

use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::F32;
use rand::distributions::Uniform;
use rand_distr::Normal;

use test::Bencher;

#[bench]
fn uniform_f32(b: &mut Bencher) {
    let m = 100;
    b.iter(|| Array::random((m, m), Uniform::new(-1f32, 1.)));
}

#[bench]
fn norm_f32(b: &mut Bencher) {
    let m = 100;
    b.iter(|| Array::random((m, m), F32(Normal::new(0., 1.))));
}

#[bench]
fn norm_f64(b: &mut Bencher) {
    let m = 100;
    b.iter(|| Array::random((m, m), Normal::new(0., 1.)));
}
