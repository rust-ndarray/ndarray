#![feature(test)]
extern crate test;
use test::{black_box, Bencher};
use ndarray::{Array3, ShapeBuilder, Zip};

pub fn zip_copy(data: &Array3<f32>, out: &mut Array3<f32>) {
    Zip::from(data).and(out).apply(|&i, o| {
        *o = i;
    });
}

pub fn zip_indexed(data: &Array3<f32>, out: &mut Array3<f32>) {
    Zip::indexed(data).and(out).apply(|idx, &i, o| {
        *o = i;
    });
}

pub fn zip_mut_with(data: &Array3<f32>, out: &mut Array3<f32>) {
    out.zip_mut_with(&data, |o, &i| {
        *o = i;
    });
}

// array size in benchmarks
const SZ3: (usize, usize, usize) = (137, 171, 151);

#[bench]
fn zip_cf(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim().f());
    b.iter(|| black_box(zip_copy(&data, &mut out)));
}

#[bench]
fn zip_cc(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim());
    b.iter(|| black_box(zip_copy(&data, &mut out)));
}

#[bench]
fn zip_fc(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim());
    b.iter(|| black_box(zip_copy(&data, &mut out)));
}

#[bench]
fn zip_ff(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim().f());
    b.iter(|| black_box(zip_copy(&data, &mut out)));
}

#[bench]
fn zip_indexed_cf(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim().f());
    b.iter(|| black_box(zip_indexed(&data, &mut out)));
}

#[bench]
fn zip_indexed_cc(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim());
    b.iter(|| black_box(zip_indexed(&data, &mut out)));
}

#[bench]
fn zip_indexed_fc(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim());
    b.iter(|| black_box(zip_indexed(&data, &mut out)));
}

#[bench]
fn zip_indexed_ff(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim().f());
    b.iter(|| black_box(zip_indexed(&data, &mut out)));
}

#[bench]
fn zip_mut_with_cf(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim().f());
    b.iter(|| black_box(zip_mut_with(&data, &mut out)));
}

#[bench]
fn zip_mut_with_cc(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim());
    b.iter(|| black_box(zip_mut_with(&data, &mut out)));
}

#[bench]
fn zip_mut_with_fc(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim());
    b.iter(|| black_box(zip_mut_with(&data, &mut out)));
}

#[bench]
fn zip_mut_with_ff(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim().f());
    b.iter(|| black_box(zip_mut_with(&data, &mut out)));
}
