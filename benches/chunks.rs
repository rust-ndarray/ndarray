#![feature(test)]

extern crate test;
use test::Bencher;

#[macro_use(azip)]
extern crate ndarray;
use ndarray::prelude::*;
use ndarray::NdProducer;

#[bench]
fn chunk2x2_sum(bench: &mut Bencher)
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
