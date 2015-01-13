
extern crate num;
extern crate ndarray;

use ndarray::{arr1, arr2};
use num::{Num, Complex};

fn c<T: Clone + Num>(re: T, im: T) -> Complex<T> {
    Complex::new(re, im)
}

#[test]
fn complex_mat_mul()
{
    let a = arr2(&[&[c::<f32>(1., 1.), c(2., 0.)], &[c(0., -2.), c(3., 0.)]]);
    let e = ndarray::linalg::eye(2);
    let r = a.mat_mul(&e);
    println!("{:?}", a);
    assert_eq!(r, a);
    assert_eq!(a.mean(0), arr1(&[c(0.5, -0.5), c(2.5, 0.)]));
}
