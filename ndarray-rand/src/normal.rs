//! Implementation of the multiavariate normal distribution.
use crate::RandomExt;
use ndarray::{Array, IntoDimension, Dimension};
use crate::rand::Rng;
use crate::rand::distributions::Distribution;
use crate::rand_distr::{StandardNormal};

#[cfg(feature = "normaldist")]
pub mod advanced;

/// Standard multivariate normal distribution `N(0,1)` for any-dimensional arrays.
///
/// ```
/// use rand;
/// use rand_distr::Distribution;
/// use ndarray;
/// use ndarray_rand::normal::MultivariateStandardNormal;
/// 
/// let shape = (2, 3); // create (2,3)-matrix of standard normal variables
/// let n = MultivariateStandardNormal::new(shape);
/// let ref mut rng = rand::thread_rng();
/// println!("{:?}", n.sample(rng));
/// ```
pub struct MultivariateStandardNormal<D>
where D: Dimension
{
    shape: D
}

impl<D> MultivariateStandardNormal<D>
where D: Dimension
{
    pub fn new<Sh>(shape: Sh) -> Self
    where Sh: IntoDimension<Dim=D>
    {
        MultivariateStandardNormal {
            shape: shape.into_dimension()
        }
    }

    pub fn shape(&self) -> D {
        self.shape.clone()
    }
}

impl<D> Distribution<Array<f64, D>> for MultivariateStandardNormal<D>
where D: Dimension
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array<f64, D> {
        let shape = self.shape.clone();
        let res = Array::random_using(
            shape, StandardNormal, rng);
        res
    }
}
