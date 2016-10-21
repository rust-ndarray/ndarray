#![feature(test)]

extern crate test;
use test::Bencher;

#[macro_use(s)]
extern crate ndarray;
use ndarray::prelude::*;

#[bench]
fn default_f64(bench: &mut Bencher) {
    bench.iter(|| {
        Array::<f64, _>::default((128, 128))
    })
}

#[bench]
fn zeros_f64(bench: &mut Bencher) {
    bench.iter(|| {
        Array::<f64, _>::zeros((128, 128))
    })
}

#[bench]
fn map_regular(bench: &mut test::Bencher)
{
    let a = Array::linspace(0., 127., 128).into_shape((8, 16)).unwrap();
    bench.iter(|| {
        a.map(|&x| 2. * x)
    });
}


#[bench]
fn map_stride(bench: &mut test::Bencher)
{
    let a = Array::linspace(0., 127., 256).into_shape((8, 32)).unwrap();
    let av = a.slice(s![.., ..;2]);
    bench.iter(|| {
        av.map(|&x| 2. * x)
    });
}
