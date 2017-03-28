
#[macro_use]
extern crate ndarray;

use ndarray::prelude::*;
use ndarray::Zip;
use std::marker::PhantomData;

fn main() {
    let mut a = <Array1<f32>>::linspace(1., 100., 10 * 10).into_shape((10, 10)).unwrap();
    let at = a.t();
    let iter = at.whole_chunks((5, 4));
    for elt in iter {
        println!("{:6.2?}", elt);
    }
    let iter = at.whole_chunks((5, 4));
    /*
    */
    let mut b = <Array2<f32>>::zeros((2, 2));
    println!("{:?}", 
        Zip::from(&mut b).and(iter.clone())
    );

    Zip::from(&mut b).and(iter).apply(|b, a| {
        println!("{:6.2?}", a);
        *b = a.row(0).scalar_sum();
    });
    println!("{:?}", b);
    Zip::from(b.view_mut().reversed_axes()).and(a.whole_chunks([4, 5])).apply(|b, a| {
        println!("{:6.2?}", a);
        *b = a.row(0).scalar_sum();
    });
    println!("{:?}", b);
    //array_zip!(mut a (a), mut b in { *b = a.scalar_sum() });
}
