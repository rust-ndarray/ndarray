#![feature(test)]

extern crate test;

use ndarray::Array;
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use rand_distr::Uniform;

use test::Bencher;

#[bench]
fn uniform_f32(b: &mut Bencher)
{
    let m = 100;
    b.iter(|| Array::random((m, m), Uniform::new(-1f32, 1.).unwrap()));
}

#[bench]
#[cfg(feature = "half")]
fn norm_f16(b: &mut Bencher)
{
    let m = 100;
    b.iter(|| Array::random((m, m), Normal::new(half::f16::ZERO, half::f16::ONE).unwrap()));
}

#[bench]
#[cfg(feature = "half")]
fn norm_bf16(b: &mut Bencher)
{
    let m = 100;
    b.iter(|| Array::random((m, m), Normal::new(half::bf16::ZERO, half::bf16::ONE).unwrap()));
}

#[bench]
fn norm_f32(b: &mut Bencher)
{
    let m = 100;
    b.iter(|| Array::random((m, m), Normal::new(0f32, 1.).unwrap()));
}

#[bench]
fn norm_f64(b: &mut Bencher)
{
    let m = 100;
    b.iter(|| Array::random((m, m), Normal::new(0f64, 1.).unwrap()));
}
