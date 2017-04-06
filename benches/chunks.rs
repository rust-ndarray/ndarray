#![feature(test)]

extern crate test;
use test::Bencher;

#[macro_use(azip)]
extern crate ndarray;
use ndarray::prelude::*;
use ndarray::NdProducer;

#[bench]
fn chunk2x2_iter_sum(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((256, 256));
    let chunksz = (2, 2);
    let mut sum = Array::zeros(a.whole_chunks(chunksz).raw_dim());
    bench.iter(|| {
        azip!(ref a (a.whole_chunks(chunksz)), mut sum in {
            *sum = a.iter().sum::<f32>();
        });
    });
}

#[bench]
fn chunk2x2_scalar_sum(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((256, 256));
    let chunksz = (2, 2);
    let mut sum = Array::zeros(a.whole_chunks(chunksz).raw_dim());
    bench.iter(|| {
        azip!(ref a (a.whole_chunks(chunksz)), mut sum in {
            *sum = a.scalar_sum();
        });
    });
}

#[bench]
fn chunk2x2_sum_get1(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((256, 256));
    let chunksz = (2, 2);
    let mut sum = Array::<f32, _>::zeros(a.whole_chunks(chunksz).raw_dim());
    bench.iter(|| {
        let (m, n) = a.dim();
        for i in 0..m {
            for j in 0..n {
                sum[[i/2, j/2]] += a[[i, j]];
            }
        }
    });
}

#[bench]
fn chunk2x2_sum_uget1(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((256, 256));
    let chunksz = (2, 2);
    let mut sum = Array::<f32, _>::zeros(a.whole_chunks(chunksz).raw_dim());
    bench.iter(|| {
        let (m, n) = a.dim();
        for i in 0..m {
            for j in 0..n {
                unsafe {
                    *sum.uget_mut([i/2, j/2]) += *a.uget([i, j]);
                }
            }
        }
    });
}

#[bench]
fn chunk2x2_sum_get2(bench: &mut Bencher)
{
    let a = Array::<f32, _>::zeros((256, 256));
    let chunksz = (2, 2);
    let mut sum = Array::<f32, _>::zeros(a.whole_chunks(chunksz).raw_dim());
    bench.iter(|| {
        let (m, n) = sum.dim();
        for i in 0..m {
            for j in 0..n {
                sum[[i, j]] += a[[i*2 + 0, j*2 + 0]];
                sum[[i, j]] += a[[i*2 + 0, j*2 + 1]];
                sum[[i, j]] += a[[i*2 + 1, j*2 + 1]];
                sum[[i, j]] += a[[i*2 + 1, j*2 + 0]];
            }
        }
    });
}
