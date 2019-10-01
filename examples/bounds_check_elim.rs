#![crate_type = "lib"]
#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names
)]

// Test cases for bounds check elimination

use ndarray::prelude::*;

/*
pub fn testslice(a: &[f64]) -> f64 {
    let mut sum = 0.;
    for i in 0..a.len() {
        sum += a[i];
    }
    sum
}

pub fn testvec(a: &Vec<f64>) -> f64 {
    let mut sum = 0.;
    for i in 0..a.len() {
        sum += a[i];
    }
    sum
}

pub fn testvec_as_slice(a: &Vec<f64>) -> f64 {
    let a = a.as_slice();
    let mut sum = 0.;
    for i in 0..a.len() {
        sum += a[i];
    }
    sum
}
*/

#[no_mangle]
pub fn test1d_single(a: &Array1<f64>, i: usize) -> f64 {
    if i < a.len() {
        a[i]
    } else {
        0.
    }
}

#[no_mangle]
pub fn test1d_single_mut(a: &mut Array1<f64>, i: usize) -> f64 {
    if i < a.len() {
        *&mut a[i]
    } else {
        0.
    }
}

#[no_mangle]
pub fn test1d_len_of(a: &Array1<f64>) -> f64 {
    let a = &*a;
    let mut sum = 0.;
    for i in 0..a.len_of(Axis(0)) {
        sum += a[i];
    }
    sum
}

#[no_mangle]
pub fn test1d_range(a: &Array1<f64>) -> f64 {
    let mut sum = 0.;
    for i in 0..a.len() {
        sum += a[i];
    }
    sum
}

#[no_mangle]
pub fn test1d_while(a: &Array1<f64>) -> f64 {
    let mut sum = 0.;
    let mut i = 0;
    while i < a.len() {
        sum += a[i];
        i += 1;
    }
    sum
}

#[no_mangle]
pub fn test2d_ranges(a: &Array2<f64>) -> f64 {
    let mut sum = 0.;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            sum += a[[i, j]];
        }
    }
    sum
}

#[no_mangle]
pub fn test2d_whiles(a: &Array2<f64>) -> f64 {
    let mut sum = 0.;
    let mut i = 0;
    while i < a.nrows() {
        let mut j = 0;
        while j < a.ncols() {
            sum += a[[i, j]];
            j += 1;
        }
        i += 1;
    }
    sum
}

fn main() {}
