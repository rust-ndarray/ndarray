
//! Parallelization features for ndarray.
//!
//! **Requires crate feature `"rayon"`**
//!
//! The array views and references to owned arrays all implement
//! `IntoParallelIterator`; the default parallel iterators (each element by
//! reference or mutable reference) have no ordering guarantee in their parallel
//! implementations.
//!
//! `.axis_iter()` and `.axis_iter_mut()` also have parallel counterparts.
//!
//! # Examples
//!
//! Compute the exponential of each element in an array, parallelized.
//!
//! ```
//! use ndarray::Array2;
//! use ndarray::parallel::rayon_prelude::*;
//!
//! let mut a = Array2::<f64>::zeros((128, 128));
//! a.par_iter_mut().for_each(|x| *x = x.exp());
//! ```
//!
//! Use the parallel `.axis_iter()` to compute the sum of each row.
//!
//! ```
//! use ndarray::Array;
//! use ndarray::Axis;
//! use ndarray::parallel::rayon_prelude::*;
//!
//! let a = Array::linspace(0., 63., 64).into_shape((4, 16)).unwrap();
//! let mut sums = Vec::new();
//! a.axis_iter(Axis(0))
//!  .into_par_iter()
//!  .map(|row| row.scalar_sum())
//!  .collect_into(&mut sums);
//!
//! assert_eq!(sums, [120., 376., 632., 888.]);
//! ```

pub use rayon::prelude as rayon_prelude;
pub use iterators::par::Parallel;
