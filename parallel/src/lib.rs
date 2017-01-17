//! Parallelization features for ndarray.
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
//! extern crate ndarray;
//! extern crate ndarray_parallel;
//!
//! use ndarray::Array2;
//! use ndarray_parallel::prelude::*;
//!
//! fn main() {
//!     let mut a = Array2::<f64>::zeros((128, 128));
//!     a.par_iter_mut().for_each(|x| *x = x.exp());
//! }
//! ```
//!
//! Use the parallel `.axis_iter()` to compute the sum of each row.
//!
//! ```
//! extern crate ndarray;
//! extern crate ndarray_parallel;
//!
//! use ndarray::Array;
//! use ndarray::Axis;
//! use ndarray_parallel::prelude::*;
//!
//! fn main() {
//!     let a = Array::linspace(0., 63., 64).into_shape((4, 16)).unwrap();
//!     let mut sums = Vec::new();
//!     a.axis_iter(Axis(0))
//!      .into_par_iter()
//!      .map(|row| row.scalar_sum())
//!      .collect_into(&mut sums);
//!
//!     assert_eq!(sums, [120., 376., 632., 888.]);
//! }
//! ```


extern crate ndarray;
extern crate rayon;

pub mod prelude {
    // happy and insane; ignorance is bluss
    pub use NdarrayIntoParallelIterator;
    pub use NdarrayIntoParallelRefIterator;
    pub use NdarrayIntoParallelRefMutIterator;

    #[doc(no_inline)]
    pub use rayon::prelude::{ParallelIterator, ExactParallelIterator};
}

pub use par::Parallel;
pub use into_traits::{
    NdarrayIntoParallelIterator,
    NdarrayIntoParallelRefIterator,
    NdarrayIntoParallelRefMutIterator,
};

mod par;
mod into_traits;
mod into_impls;
