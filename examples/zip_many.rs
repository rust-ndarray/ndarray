
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

    {
        let a = a.view_mut().reversed_axes();
        azip!(mut a (a), b (b.t()) in { *a = b });

    }
    assert_eq!(a, b);

    azip!(mut a, b, c in { *a = b + c; });
    assert_eq!(a, &b + &c);
    
    // sum of each row
    let ax = Axis(0);
    let mut sums = Array::zeros(a.len_of(ax));
    azip!(mut sums, ref a (a.axis_iter(ax)) in { *sums = a.scalar_sum() });

    // sum of each chunk
    let chunk_sz = (2, 2);
    let nchunks = (n / chunk_sz.0, n / chunk_sz.1);
    let mut sums = Array::zeros(nchunks);
    azip!(mut sums, ref a (a.exact_chunks(chunk_sz)) in { *sums = a.scalar_sum() });


    // Let's imagine we split to parallelize
    {
        let (x, y) = Zip::indexed(&mut a).split();
        x.apply(|(_, j), elt| {
            *elt = elt.powi(j as i32);
        });

        y.apply(|(_, j), elt| {
            *elt = elt.powi(j as i32);
        });
    }
    println!("{:8.3?}", a);
}
