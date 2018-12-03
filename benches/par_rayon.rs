#![cfg(feature="rayon")]
#![feature(test)]

extern crate rayon;

extern crate ndarray;
extern crate itertools;

use ndarray::prelude::*;
use ndarray::parallel::prelude::*;

extern crate test;
use test::Bencher;

use ndarray::Zip;

const EXP_N: usize = 256;
const ADDN: usize = 512;

use std::cmp::max;

fn set_threads() {
    // Consider setting a fixed number of threads here, for example to avoid
    // oversubscribing on hyperthreaded cores.
    // let n = 4;
    // let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
}

#[bench]
fn map_exp_regular(bench: &mut Bencher)
{
    let mut a = Array2::<f64>::zeros((EXP_N, EXP_N));
    a.swap_axes(0, 1);
    bench.iter(|| {
        a.mapv_inplace(|x| x.exp());
    });
}

#[bench]
fn rayon_exp_regular(bench: &mut Bencher)
{
    set_threads();
    let mut a = Array2::<f64>::zeros((EXP_N, EXP_N));
    a.swap_axes(0, 1);
    bench.iter(|| {
        a.view_mut().into_par_iter().for_each(|x| *x = x.exp());
    });
}

const FASTEXP: usize = EXP_N;

#[inline]
fn fastexp(x: f64) -> f64 {
    let x = 1. + x/1024.;
    x.powi(1024)
}

#[bench]
fn map_fastexp_regular(bench: &mut Bencher)
{
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        a.mapv_inplace(|x| fastexp(x))
    });
}

#[bench]
fn rayon_fastexp_regular(bench: &mut Bencher)
{
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        a.view_mut().into_par_iter().for_each(|x| *x = fastexp(*x));
    });
}

#[bench]
fn map_fastexp_cut(bench: &mut Bencher)
{
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    let mut a = a.slice_mut(s![.., ..-1]);
    bench.iter(|| {
        a.mapv_inplace(|x| fastexp(x))
    });
}

#[bench]
fn rayon_fastexp_cut(bench: &mut Bencher)
{
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    let mut a = a.slice_mut(s![.., ..-1]);
    bench.iter(|| {
        a.view_mut().into_par_iter().for_each(|x| *x = fastexp(*x));
    });
}

#[bench]
fn map_fastexp_by_axis(bench: &mut Bencher)
{
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        for mut sheet in a.axis_iter_mut(Axis(0)) {
            sheet.mapv_inplace(fastexp)
        }
    });
}

#[bench]
fn rayon_fastexp_by_axis(bench: &mut Bencher)
{
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        a.axis_iter_mut(Axis(0)).into_par_iter()
            .for_each(|mut sheet| sheet.mapv_inplace(fastexp));
    });
}

#[bench]
fn rayon_fastexp_zip(bench: &mut Bencher)
{
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        Zip::from(&mut a).into_par_iter().for_each(|(elt, )| *elt = fastexp(*elt));
    });
}

#[bench]
fn add(bench: &mut Bencher)
{
    let mut a = Array2::<f64>::zeros((ADDN, ADDN));
    let b = Array2::<f64>::zeros((ADDN, ADDN));
    let c = Array2::<f64>::zeros((ADDN, ADDN));
    let d = Array2::<f64>::zeros((ADDN, ADDN));
    bench.iter(|| {
        azip!(mut a, b, c, d in {
            *a += b.exp() + c.exp() + d.exp();
        });
    });
}

#[bench]
fn rayon_add(bench: &mut Bencher)
{
    set_threads();
    let mut a = Array2::<f64>::zeros((ADDN, ADDN));
    let b = Array2::<f64>::zeros((ADDN, ADDN));
    let c = Array2::<f64>::zeros((ADDN, ADDN));
    let d = Array2::<f64>::zeros((ADDN, ADDN));
    bench.iter(|| {
        par_azip!(mut a, b, c, d in {
            *a += b.exp() + c.exp() + d.exp();
        });
    });
}
