
#[macro_use]
extern crate ndarray;

use ndarray::prelude::*;
use ndarray::Zip;

macro_rules! array_zip {
    ($($x:pat),* in ($a:expr, $($array:expr),*) { $($t:tt)* }) => {
        Zip::from($a)
         $(
             .and($array)
         )*
        .apply(|$($x),*| {
            $($t)*
        })
    }
}

fn main() {
    let n = 16;
    let mut a = Array::<f32, _>::zeros((n, n));
    let mut b = Array::<f32, _>::from_elem((n, n), 1.);
    for ((i, j), elt) in b.indexed_iter_mut() {
        *elt /= 1. + (i + j) as f32;
    }
    let c = Array::<f32, _>::from_elem((n, n + 1), 1.7);
    let c = c.slice(s![.., ..-1]);

    let d = Array::from_elem((1, n), 1.);
    let e = Array::from_elem((), 2.);

    {
        let mut z = Zip::from(a.view_mut()).and_broadcast(&d).and_broadcast(&e);
        z.apply(|x, &y, &z| *x = y + z);
    }
    assert!(a.iter().all(|&x| x == 3.));

    {
        let mut z = Zip::from(b.t()).and(a.view_mut().reversed_axes());
        z.apply(|&x, y| *y = x);

    }
    assert_eq!(a, b);

    array_zip!(x, &y, &z in (&mut a, &b, &c) {
        *x = y + z;
    });
    assert_eq!(a, &b + &c);

    a.fill(0.);
    for _ in 0..10_000 {
        array_zip!(x, &y, &z in (&mut a, &b, &c) {
            *x += y * z;
        });
    }
    println!("{:4.2?}", a);
}
