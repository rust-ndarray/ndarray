#![feature(test)]
#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names
)]
extern crate test;
use test::Bencher;

use ndarray::prelude::*;
use ndarray::ErrorKind;

// Use ZST elements to remove allocation from the benchmarks

#[derive(Copy, Clone, Debug)]
struct Zst;

type A4 = Array4<Zst>;

#[bench]
fn from_elem(bench: &mut Bencher) {
    bench.iter(|| {
        A4::from_elem((1, 2, 3, 4), Zst)
    })
}

#[bench]
fn from_shape_vec_ok(bench: &mut Bencher) {
    bench.iter(|| {
        let v: Vec<Zst> = vec![Zst; 1 * 2 * 3 * 4];
        let x = A4::from_shape_vec((1, 2, 3, 4).strides((24, 12, 4, 1)), v);
        debug_assert!(x.is_ok(), "problem with {:?}", x);
        x
    })
}

#[bench]
fn from_shape_vec_fail(bench: &mut Bencher) {
    bench.iter(|| {
        let v: Vec<Zst> = vec![Zst; 1 * 2 * 3 * 4];
        let x = A4::from_shape_vec((1, 2, 3, 4).strides((4, 3, 2, 1)), v);
        debug_assert!(x.is_err());
        x
    })
}

#[bench]
fn into_shape_fail(bench: &mut Bencher) {
    let a = A4::from_elem((1, 2, 3, 4), Zst);
    let v = a.view();
    bench.iter(|| {
        v.clone().into_shape((5, 3, 2, 1))
    })
}

#[bench]
fn into_shape_ok_c(bench: &mut Bencher) {
    let a = A4::from_elem((1, 2, 3, 4), Zst);
    let v = a.view();
    bench.iter(|| {
        v.clone().into_shape((4, 3, 2, 1))
    })
}

#[bench]
fn into_shape_ok_f(bench: &mut Bencher) {
    let a = A4::from_elem((1, 2, 3, 4).f(), Zst);
    let v = a.view();
    bench.iter(|| {
        v.clone().into_shape((4, 3, 2, 1))
    })
}

#[bench]
fn stack_ok(bench: &mut Bencher) {
    let a = Array::from_elem((15, 15), Zst);
    let rows = a.rows().into_iter().collect::<Vec<_>>();
    bench.iter(|| {
        let res = ndarray::stack(Axis(1), &rows);
        debug_assert!(res.is_ok(), "err {:?}", res);
        res
    });
}

#[bench]
fn stack_err_axis(bench: &mut Bencher) {
    let a = Array::from_elem((15, 15), Zst);
    let rows = a.rows().into_iter().collect::<Vec<_>>();
    bench.iter(|| {
        let res = ndarray::stack(Axis(2), &rows);
        debug_assert!(res.is_err());
        res
    });
}

#[bench]
fn stack_err_shape(bench: &mut Bencher) {
    let a = Array::from_elem((15, 15), Zst);
    let rows = a.rows().into_iter()
        .enumerate()
        .map(|(i, mut row)| { row.slice_collapse(s![..(i as isize)]); row })
        .collect::<Vec<_>>();
    bench.iter(|| {
        let res = ndarray::stack(Axis(1), &rows);
        debug_assert!(res.is_err());
        debug_assert_eq!(res.clone().unwrap_err().kind(), ErrorKind::IncompatibleShape);
        res
    });
}
