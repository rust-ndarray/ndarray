//! Implementation of the multiavariate normal distribution.
use crate::RandomExt;
use ndarray::Ix1;
use ndarray::Array1;
use rand::Rng;
use rand::distributions::{Distribution, StandardNormal};

pub mod advanced;

pub struct MultivariateStandardNormal
{
    shape: Ix1
}

/// Standard multivariate normal distribution `N(0,1)`.
///
/// ```
/// use ndarray;
/// use ndarray_rand::normal::MultivariateStandardNormal;
///
/// let n = MultivariateStandardNormal();
/// ```
impl MultivariateStandardNormal
{
    pub fn new(shape: Ix1) -> Self {
        MultivariateStandardNormal {
            shape
        }
    }
}

impl Distribution<Array1<f64>> for MultivariateStandardNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
        let shape = self.shape.clone();
        let res = Array1::random_using(
            shape, StandardNormal, rng);
        res
    }
}
