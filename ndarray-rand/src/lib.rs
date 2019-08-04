// Copyright 2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Constructors for randomized arrays: `rand` integration for `ndarray`.
//!
//! See [**`RandomExt`**](trait.RandomExt.html) for usage examples.
//!
//! ## Note
//!
//! `ndarray-rand` depends on [`rand` 0.7.0](https://docs.rs/rand/0.7.0/rand/).
//!
//! [`rand`](https://docs.rs/rand/0.7.0/rand/) and [`rand-distr`](https://docs.rs/rand_distr/0.2.1/rand_distr/)
//! are re-exported as sub-modules, [`ndarray_rand::rand`](rand/index.html)
//! and [`ndarray_rand::rand_distr`](rand_distr/index.html) respectively.
//! Please rely on these submodules for guaranteed version compatibility.
//!
//! If you want to use a random number generator or distribution from another crate
//! with `ndarray-rand`, you need to make sure that the other crate also depends on the
//! same version of `rand`. Otherwise, the compiler will return errors saying
//! that the items are not compatible (e.g. that a type doesn't implement a
//! necessary trait).


use crate::rand::rngs::SmallRng;
use crate::rand::{thread_rng, Rng, SeedableRng};
use crate::rand::distributions::Distribution;

use ndarray::ShapeBuilder;
use ndarray::{ArrayBase, DataOwned, Dimension};

/// [`rand`](https://docs.rs/rand/0.7.0/rand/), re-exported for convenience and version-compatibility.
pub mod rand {
    pub use rand::*;
}

/// [`rand-distr`](https://docs.rs/rand_distr/0.2.1/rand_distr/), re-exported for convenience and version-compatibility.
pub mod rand_distr {
    pub use rand_distr::*;
}

/// Constructors for n-dimensional arrays with random elements.
///
/// This trait extends ndarray’s `ArrayBase` and can not be implemented
/// for other types.
///
/// The default RNG is a fast automatically seeded rng (currently
/// [`rand::rngs::SmallRng`](https://docs.rs/rand/0.7/rand/rngs/struct.SmallRng.html)
/// seeded from [`rand::thread_rng`](https://docs.rs/rand/0.7/rand/fn.thread_rng.html)).
///
/// Note that `SmallRng` is cheap to initialize and fast, but it may generate
/// low-quality random numbers, and reproducibility is not guaranteed. See its
/// documentation for information. You can select a different RNG with
/// [`.random_using()`](#tymethod.random_using).
pub trait RandomExt<S, D>
where
    S: DataOwned,
    D: Dimension,
{
    /// Create an array with shape `dim` with elements drawn from
    /// `distribution` using the default RNG.
    ///
    /// ***Panics*** if creation of the RNG fails or if the number of elements
    /// overflows usize.
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate ndarray_rand;
    ///
    /// use ndarray::Array;
    /// use ndarray_rand::RandomExt;
    /// use ndarray_rand::rand_distr::Uniform;
    ///
    /// # fn main() {
    /// let a = Array::random((2, 5), Uniform::new(0., 10.));
    /// println!("{:8.4}", a);
    /// // Example Output:
    /// // [[  8.6900,   6.9824,   3.8922,   6.5861,   2.4890],
    /// //  [  0.0914,   5.5186,   5.8135,   5.2361,   3.1879]]
    /// # }
    fn random<Sh, IdS>(shape: Sh, distribution: IdS) -> ArrayBase<S, D>
    where
        IdS: Distribution<S::Elem>,
        Sh: ShapeBuilder<Dim = D>;

    /// Create an array with shape `dim` with elements drawn from
    /// `distribution`, using a specific Rng `rng`.
    ///
    /// ***Panics*** if the number of elements overflows usize.
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate ndarray_rand;
    /// extern crate rand_isaac;
    ///
    /// use ndarray::Array;
    /// use ndarray_rand::RandomExt;
    /// use ndarray_rand::rand::SeedableRng;
    /// use ndarray_rand::rand_distr::Uniform;
    /// use rand_isaac::isaac64::Isaac64Rng;
    ///
    /// # fn main() {
    /// // Get a seeded random number generator for reproducibility (Isaac64 algorithm)
    /// let seed = 42;
    /// let mut rng = Isaac64Rng::seed_from_u64(seed);
    ///
    /// // Generate a random array using `rng`
    /// let a = Array::random_using((2, 5), Uniform::new(0., 10.), &mut rng);
    /// println!("{:8.4}", a);
    /// // Example Output:
    /// // [[  8.6900,   6.9824,   3.8922,   6.5861,   2.4890],
    /// //  [  0.0914,   5.5186,   5.8135,   5.2361,   3.1879]]
    /// # }
    fn random_using<Sh, IdS, R>(shape: Sh, distribution: IdS, rng: &mut R) -> ArrayBase<S, D>
    where
        IdS: Distribution<S::Elem>,
        R: Rng + ?Sized,
        Sh: ShapeBuilder<Dim = D>;
}

impl<S, D> RandomExt<S, D> for ArrayBase<S, D>
where
    S: DataOwned,
    D: Dimension,
{
    fn random<Sh, IdS>(shape: Sh, dist: IdS) -> ArrayBase<S, D>
    where
        IdS: Distribution<S::Elem>,
        Sh: ShapeBuilder<Dim = D>,
    {
        let mut rng =
            SmallRng::from_rng(thread_rng()).expect("create SmallRng from thread_rng failed");
        Self::random_using(shape, dist, &mut rng)
    }

    fn random_using<Sh, IdS, R>(shape: Sh, dist: IdS, rng: &mut R) -> ArrayBase<S, D>
    where
        IdS: Distribution<S::Elem>,
        R: Rng + ?Sized,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_shape_fn(shape, |_| dist.sample(rng))
    }
}

/// A wrapper type that allows casting f64 distributions to f32
///
/// ```
/// extern crate ndarray;
/// extern crate ndarray_rand;
///
/// use ndarray::Array;
/// use ndarray_rand::{RandomExt, F32};
/// use ndarray_rand::rand_distr::Normal;
///
/// # fn main() {
/// let distribution_f64 = Normal::new(0., 1.).expect("Failed to create normal distribution");
/// let a = Array::random((2, 5), F32(distribution_f64));
/// println!("{:8.4}", a);
/// // Example Output:
/// // [[ -0.6910,   1.1730,   1.0902,  -0.4092,  -1.7340],
/// //  [ -0.6810,   0.1678,  -0.9487,   0.3150,   1.2981]]
/// # }
#[derive(Copy, Clone, Debug)]
pub struct F32<S>(pub S);

impl<S> Distribution<f32> for F32<S>
where
    S: Distribution<f64>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
        self.0.sample(rng) as f32
    }
}
