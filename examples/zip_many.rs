
#[macro_use]
extern crate ndarray;

use ndarray::prelude::*;
use ndarray::Zip;

fn main() {
    let n = 16;
    let mut a = Array::<f32, _>::zeros((n, n));
    let mut b = Array::<f32, _>::from_elem((n, n), 1.);
    for ((i, j), elt) in b.indexed_iter_mut() {
        *elt /= 1. + (i + 2 * j) as f32;
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
        let a = a.view_mut().reversed_axes();
        array_zip!(mut a (a), b (b.t()) in { *a = b });

    }
    assert_eq!(a, b);

    array_zip!(mut a, b, c in { *a = b + c; });
    assert_eq!(a, &b + &c);

    let ax = Axis(0);
    println!("{:?}", Zip::from(b.row_mut(0)).and(a.axis_iter(ax)));
    Zip::from(b.row_mut(0)).and(a.axis_iter(ax)).apply(|x, y| {
        println!("{:6.2?}", y);
        *x = y.scalar_sum();
    });
    //array_zip!(a (a.axis_iter(Axis(0))), b (b.column(0)) in { } );

    a.fill(0.);
    for _ in 0..10_000 {
        array_zip!(mut a, b, c in { *a += b * c });
    }
    println!("{:8.2?}", a);
}
