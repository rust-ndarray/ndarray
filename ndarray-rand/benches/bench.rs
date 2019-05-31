#![feature(test)]

extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate test;

use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::F32;
use rand::distributions::Normal;
use rand::distributions::Uniform;

use test::Bencher;

#[bench]
fn uniform_f32(b: &mut Bencher) {
    let m = 100;
    b.iter(|| {
        let a = Array::random((m, m), Uniform::new(-1f32, 1.));
        a
    });
}

#[bench]
fn norm_f32(b: &mut Bencher) {
    let m = 100;
    b.iter(|| {
        let a = Array::random((m, m), F32(Normal::new(0., 1.)));
        a
    });
}

#[bench]
fn norm_f64(b: &mut Bencher) {
    let m = 100;
    b.iter(|| {
        let a = Array::random((m, m), Normal::new(0., 1.));
        a
    });
}
