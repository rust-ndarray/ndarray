//! Implementation of the multiavariate normal distribution.
use super::RandomExt;
use ndarray::Ix1;
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::distributions::{Distribution, StandardNormal};

pub struct MultivariateStandardNormal
{
    shape: Ix1
}

impl MultivariateStandardNormal
{
    pub fn new(shape: Ix1) -> Self {
        MultivariateStandardNormal {
            shape
        }
    }
}

impl Distribution<Array1<f64>> for MultivariateStandardNormal
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
        let shape = self.shape.clone();
        let res = Array1::random_using(
            shape, StandardNormal, rng);
        res
    }
}

#[cfg(feature = "normal-dist")]
/// The normal distribution `N(mean, covariance)`.
pub struct MultivariateNormal {
    mean: Array1<f64>,
    covariance: Array2<f64>
}
