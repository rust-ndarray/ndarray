
extern crate num_traits;
extern crate num_complex;
extern crate ndarray;

use ndarray::{arr1, arr2, Axis};
use ndarray::Array;
use num_traits::Num;
use num_complex::Complex;

fn c<T: Clone + Num>(re: T, im: T) -> Complex<T> {
    Complex::new(re, im)
}

#[test]
fn complex_mat_mul()
{
    let a = arr2(&[[c(3., 4.), c(2., 0.)], [c(0., -2.), c(3., 0.)]]);
    let b = (&a * c(3., 0.)).map(|c| 5. * c / c.norm());
    println!("{:>8.2}", b);
    let e = Array::eye(2);
    let r = a.dot(&e);
    println!("{}", a);
    assert_eq!(r, a);
    assert_eq!(a.mean_axis(Axis(0)).unwrap(), arr1(&[c(1.5, 1.), c(2.5, 0.)]));
}
