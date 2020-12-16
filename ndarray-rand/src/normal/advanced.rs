/// The normal distribution `N(mean, covariance)`.
use rand::Rng;
use rand::distributions::{
    Distribution, StandardNormal
};

use ndarray::prelude::*;
use ndarray_linalg::error::Result as LAResult;

/// Multivariate normal distribution for 1D arrays,
/// with mean vector and covariance matrix.
pub struct MultivariateNormal {
    shape: Ix1,
    mean: Array1<f64>,
    covariance: Array2<f64>,
    /// Lower triangular matrix (Cholesky decomposition of the coviariance matrix)
    lower: Array2<f64>
}

impl MultivariateNormal {
    pub fn new(mean: Array1<f64>, covariance: Array2<f64>) -> LAResult<Self> {
        let shape: Ix1 = Ix1(mean.shape()[0]);
        use ndarray_linalg::cholesky::*;
        let lower = covariance.cholesky(UPLO::Lower)?;
        Ok(MultivariateNormal {
            shape, mean, covariance, lower
        })
    }

    pub fn shape(&self) -> Ix1 {
        self.shape
    }

    pub fn mean(&self) -> ArrayView1<f64> {
        self.mean.view()
    }

    pub fn covariance(&self) -> ArrayView2<f64> {
        self.covariance.view()
    }
}

impl Distribution<Array1<f64>> for MultivariateNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
        let shape = self.shape.clone();
        // standard normal distribution
        use crate::RandomExt;
        let res = Array1::random_using(
            shape, StandardNormal, rng);
        // use Cholesky decomposition to obtain a sample of our general multivariate normal
        self.mean.clone() + self.lower.view().dot(&res)
    }
}
