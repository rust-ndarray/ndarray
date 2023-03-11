#![feature(test)]

extern crate test;
use test::Bencher;

use ndarray::prelude::*;

#[bench]
fn push_reserve(bench: &mut Bencher)
{
    let ones: Array<f32, _> = array![1f32];
    bench.iter(|| {
        let mut a: Array<f32, Ix1> = array![];
        a.reserve(Axis(0), 100);
        for _ in 0..100 {
            a.append(Axis(0), ones.view()).unwrap();
        }
    });
}

#[bench]
fn push_no_reserve(bench: &mut Bencher)
{
    let ones: Array<f32, _> = array![1f32];
    bench.iter(|| {
        let mut a: Array<f32, Ix1> = array![];
        for _ in 0..100 {
            a.append(Axis(0), ones.view()).unwrap();
        }
    });
}
