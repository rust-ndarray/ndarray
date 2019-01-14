//! Implementation of the multiavariate normal distribution.
use super::RandomExt;
use ndarray::Ix1;
use ndarray::Array1;
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

impl Distribution<Array1<f64>> for MultivariateStandardNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
        let shape = self.shape.clone();
        let res = Array1::random_using(
            shape, StandardNormal, rng);
        res
    }
}

#[cfg(feature = "normal-dist")]
/// The normal distribution `N(mean, covariance)`.
mod advanced_normal {
    use rand::Rng;
    use rand::distributions::{
        Distribution, StandardNormal
    };
    use crate::RandomExt;
    use ndarray::{Ix1, Array1, Array2};
    use ndarray_linalg::cholesky::*;

    pub struct MultivariateNormal {
        shape: Ix1,
        mean: Array1<f64>,
        covariance: Array2<f64>
    }

    impl MultivariateNormal {
        pub fn new(mean: Array1<f64>, covariance: Array2<f64>) -> Self {
            let shape = mean.shape() as Ix1;
            assert_eq!(shape[0], covariance.shape()[0]);
            MultivariateNormal {
                shape, mean, covariance
            }
        }
    }

    impl Distribution<Array1<f64>> for MultivariateNormal {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
            let shape = self.shape.clone();
            // standard normal distribution
            let res = Array1::random_using(
                shape, StandardNormal, rng);
            // use Cholesky decomposition to obtain a sample of our general multivariate normal
            let l: Array2<f64> = self.covariance.view().cholesky(UPLO::Lower).unwrap();
            self.mean.view() + l * res
        }
    }

}