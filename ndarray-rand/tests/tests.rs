extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use rand::distributions::{
    Uniform, Distribution
};
use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::normal::MultivariateStandardNormal;

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
    let shape = 2usize;
    let n = MultivariateStandardNormal::new(shape);
    let ref mut rng = rand::thread_rng();
    let s: ndarray::Array1<f64> = n.sample(rng);
    assert_eq!(s.shape(), &[2]);
}

#[cfg(features = "normaldist")]
#[test]
fn test_normal() {
    use ndarray::IntoDimension;
    use ndarray::{Array1, arr2};
    use ndarray_rand::normal::advanced::MultivariateNormal;
    let mean = Array1::from_vec([1., 0.]);
    let covar = arr2([
        [1., 0.8], [0.8, 1.]]);
    let ref mut rng = rand::thread_rng();
    let n = MultivariateNormal::new(mean, covar);
    if let Ok(n) = n {
        let x = n.sample(rng);
        assert_eq!(x.shape(), &[2, 2]);
    }
}
