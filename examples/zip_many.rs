#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names
)]

use ndarray::prelude::*;
use ndarray::Zip;

fn main() {
    let n = 6;

    let mut a = Array::<f32, _>::zeros((n, n));
    let mut b = Array::<f32, _>::zeros((n, n));
    for ((i, j), elt) in b.indexed_iter_mut() {
        *elt = 1. / (1. + (i + 2 * j) as f32);
    }
    let c = Array::<f32, _>::from_elem((n, n + 1), 1.7);
    let c = c.slice(s![.., ..-1]);

    // Using Zip for arithmetic ops across a, b, c
    Zip::from(&mut a).and(&b).and(&c)
        .for_each(|a, &b, &c| *a = b + c);
    assert_eq!(a, &b + &c);

    // and this is how to do the *same thing* with azip!()
    azip!((a in &mut a, &b in &b, &c in c) *a = b + c);

    println!("{:8.4}", a);

    // sum of each row
    let mut sums = Array::zeros(a.nrows());
    Zip::from(a.rows()).and(&mut sums)
        .for_each(|row, sum| *sum = row.sum());
    // show sums as a column matrix
    println!("{:8.4}", sums.insert_axis(Axis(1)));

    // sum of each 2x2 chunk
    let chunk_sz = (2, 2);
    let nchunks = (n / chunk_sz.0, n / chunk_sz.1);
    let mut sums = Array::zeros(nchunks);

    Zip::from(a.exact_chunks(chunk_sz))
        .and(&mut sums)
        .for_each(|chunk, sum| *sum = chunk.sum());
    println!("{:8.4}", sums);
}
