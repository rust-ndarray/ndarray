#![cfg(feature = "rayon")]
#![feature(test)]

use ndarray::parallel::prelude::*;
use ndarray::prelude::*;

extern crate test;
use test::Bencher;

use ndarray::Zip;

const EXP_N: usize = 256;
const ADDN: usize = 512;

fn set_threads() {
    // Consider setting a fixed number of threads here, for example to avoid
    // oversubscribing on hyperthreaded cores.
    // let n = 4;
    // let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
}

#[bench]
fn map_exp_regular(bench: &mut Bencher) {
    let mut a = Array2::<f64>::zeros((EXP_N, EXP_N));
    a.swap_axes(0, 1);
    bench.iter(|| {
        a.mapv_inplace(|x| x.exp());
    });
}

#[bench]
fn rayon_exp_regular(bench: &mut Bencher) {
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
    let x = 1. + x / 1024.;
    x.powi(1024)
}

#[bench]
fn map_fastexp_regular(bench: &mut Bencher) {
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| a.mapv_inplace(|x| fastexp(x)));
}

#[bench]
fn rayon_fastexp_regular(bench: &mut Bencher) {
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        a.view_mut().into_par_iter().for_each(|x| *x = fastexp(*x));
    });
}

#[bench]
fn map_fastexp_cut(bench: &mut Bencher) {
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    let mut a = a.slice_mut(s![.., ..-1]);
    bench.iter(|| a.mapv_inplace(|x| fastexp(x)));
}

#[bench]
fn rayon_fastexp_cut(bench: &mut Bencher) {
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    let mut a = a.slice_mut(s![.., ..-1]);
    bench.iter(|| {
        a.view_mut().into_par_iter().for_each(|x| *x = fastexp(*x));
    });
}

#[bench]
fn map_fastexp_by_axis(bench: &mut Bencher) {
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        for mut sheet in a.axis_iter_mut(Axis(0)) {
            sheet.mapv_inplace(fastexp)
        }
    });
}

#[bench]
fn rayon_fastexp_by_axis(bench: &mut Bencher) {
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        a.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut sheet| sheet.mapv_inplace(fastexp));
    });
}

#[bench]
fn rayon_fastexp_zip(bench: &mut Bencher) {
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    bench.iter(|| {
        Zip::from(&mut a)
            .into_par_iter()
            .for_each(|(elt,)| *elt = fastexp(*elt));
    });
}

#[bench]
fn add(bench: &mut Bencher) {
    let mut a = Array2::<f64>::zeros((ADDN, ADDN));
    let b = Array2::<f64>::zeros((ADDN, ADDN));
    let c = Array2::<f64>::zeros((ADDN, ADDN));
    let d = Array2::<f64>::zeros((ADDN, ADDN));
    bench.iter(|| {
        azip!((a in &mut a, &b in &b, &c in &c, &d in &d) {
            *a += b.exp() + c.exp() + d.exp();
        });
    });
}

#[bench]
fn rayon_add(bench: &mut Bencher) {
    set_threads();
    let mut a = Array2::<f64>::zeros((ADDN, ADDN));
    let b = Array2::<f64>::zeros((ADDN, ADDN));
    let c = Array2::<f64>::zeros((ADDN, ADDN));
    let d = Array2::<f64>::zeros((ADDN, ADDN));
    bench.iter(|| {
        par_azip!((a in &mut a, b in &b, c in &c, d in &d) {
            *a += b.exp() + c.exp() + d.exp();
        });
    });
}

const COLL_STRING_N: usize = 64;
const COLL_F64_N: usize = 128;

#[bench]
fn vec_string_collect(bench: &mut test::Bencher) {
    let v = vec![""; COLL_STRING_N * COLL_STRING_N];
    bench.iter(|| {
        v.iter().map(|s| s.to_owned()).collect::<Vec<_>>()
    });
}

#[bench]
fn array_string_collect(bench: &mut test::Bencher) {
    let v = Array::from_elem((COLL_STRING_N, COLL_STRING_N), "");
    bench.iter(|| {
        Zip::from(&v).par_map_collect(|s| s.to_owned())
    });
}

#[bench]
fn vec_f64_collect(bench: &mut test::Bencher) {
    let v = vec![1.; COLL_F64_N * COLL_F64_N];
    bench.iter(|| {
        v.iter().map(|s| s + 1.).collect::<Vec<_>>()
    });
}

#[bench]
fn array_f64_collect(bench: &mut test::Bencher) {
    let v = Array::from_elem((COLL_F64_N, COLL_F64_N), 1.);
    bench.iter(|| {
        Zip::from(&v).par_map_collect(|s| s + 1.)
    });
}

