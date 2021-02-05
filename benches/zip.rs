#![feature(test)]
extern crate test;
use test::{black_box, Bencher};
use ndarray::{Array3, ShapeBuilder, Zip};
use ndarray::s;
use ndarray::IntoNdProducer;

pub fn zip_copy<'a, A, P, Q>(data: P, out: Q)
    where P: IntoNdProducer<Item = &'a A>,
          Q: IntoNdProducer<Item = &'a mut A, Dim = P::Dim>,
          A: Copy + 'a
{
    Zip::from(data).and(out).for_each(|&i, o| {
        *o = i;
    });
}

pub fn zip_copy_split<'a, A, P, Q>(data: P, out: Q)
    where P: IntoNdProducer<Item = &'a A>,
          Q: IntoNdProducer<Item = &'a mut A, Dim = P::Dim>,
          A: Copy + 'a
{
    let z = Zip::from(data).and(out);
    let (z1, z2) = z.split();
    let (z11, z12) = z1.split();
    let (z21, z22) = z2.split();
    let f = |&i: &A, o: &mut A| *o = i;
    z11.for_each(f);
    z12.for_each(f);
    z21.for_each(f);
    z22.for_each(f);
}

pub fn zip_indexed(data: &Array3<f32>, out: &mut Array3<f32>) {
    Zip::indexed(data).and(out).for_each(|idx, &i, o| {
        let _ = black_box(idx);
        *o = i;
    });
}

// array size in benchmarks
const SZ3: (usize, usize, usize) = (100, 110, 100);

#[bench]
fn zip_cc(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim());
    b.iter(|| zip_copy(&data, &mut out));
}

#[bench]
fn zip_cf(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim().f());
    b.iter(|| zip_copy(&data, &mut out));
}

#[bench]
fn zip_fc(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim());
    b.iter(|| zip_copy(&data, &mut out));
}

#[bench]
fn zip_ff(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim().f());
    b.iter(|| zip_copy(&data, &mut out));
}

#[bench]
fn zip_indexed_cc(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim());
    b.iter(|| zip_indexed(&data, &mut out));
}

#[bench]
fn zip_indexed_ff(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim().f());
    b.iter(|| zip_indexed(&data, &mut out));
}

#[bench]
fn slice_zip_cc(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim());
    let data = data.slice(s![1.., 1.., 1..]);
    let mut out = out.slice_mut(s![1.., 1.., 1..]);
    b.iter(|| zip_copy(&data, &mut out));
}

#[bench]
fn slice_zip_ff(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim().f());
    let data = data.slice(s![1.., 1.., 1..]);
    let mut out = out.slice_mut(s![1.., 1.., 1..]);
    b.iter(|| zip_copy(&data, &mut out));
}

#[bench]
fn slice_split_zip_cc(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim());
    let data = data.slice(s![1.., 1.., 1..]);
    let mut out = out.slice_mut(s![1.., 1.., 1..]);
    b.iter(|| zip_copy_split(&data, &mut out));
}

#[bench]
fn slice_split_zip_ff(b: &mut Bencher) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim().f());
    let data = data.slice(s![1.., 1.., 1..]);
    let mut out = out.slice_mut(s![1.., 1.., 1..]);
    b.iter(|| zip_copy_split(&data, &mut out));
}
