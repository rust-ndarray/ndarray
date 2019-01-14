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
    use ndarray::prelude::*;
    use ndarray::IntoDimension;
    use ndarray_linalg::*;
    use ndarray_linalg::error::Result as LAResult;

    /// Full multivariate normal distribution, with mean vector and covariance matrix.
    pub struct MultivariateNormal {
        shape: Ix1,
        mean: Array1<f64>,
        covariance: Array2<f64>,
        /// Lower triangular Cholesky decomposition of the covariance matrix.
        lower_covariance: Array2<f64>
    }

    impl MultivariateNormal {
        pub fn new(mean: Array1<f64>, covariance: Array2<f64>) -> LAResult<Self> {
            use ndarray_linalg::cholesky::*;
            let shape = [mean.shape()[0]].into_dimension();
            let l = covariance.cholesky(UPLO::Lower);
            Ok(MultivariateNormal {
                shape, mean, covariance, lower_covariance: l
            })
        }
    }

    impl Distribution<Array1<f64>> for MultivariateNormal {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
            let shape = self.shape.clone();
            // standard normal distribution
            let res = Array1::random_using(
                shape, StandardNormal, rng);
            // use Cholesky decomposition to obtain a sample of our general multivariate normal
            self.mean.clone() + self.lower_covariance.view().dot(&res)
        }
    }

}