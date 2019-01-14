
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;

use rand::distributions::{
    Uniform, Distribution
};
use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::multivariatenormal::MultivariateStandardNormal;

#[test]
fn test_dim() {
    let (mm, nn) = (5, 5);
    for m in 0..mm {
        for n in 0..nn {
            let a = Array::random((m, n), Uniform::new(0., 2.));
            assert_eq!(a.shape(), &[m, n]);
            assert!(a.iter().all(|x| *x < 2.));
            assert!(a.iter().all(|x| *x >= 0.));
        }
    }
}

#[test]
fn test_standard_normal() {
    use ndarray::IntoDimension;
    let shape = (2usize,).into_dimension();
    let n = MultivariateStandardNormal::new(shape.into());
    let ref mut rng = rand::thread_rng();
    let s: ndarray::Array1<f64> = n.sample(rng);
    assert_eq!(s.shape(), &[2]);
}
