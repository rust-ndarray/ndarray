//! Parallelization features for ndarray.
//!
//! The array views and references to owned arrays all implement
//! `NdarrayIntoParallelIterator` (*); the default parallel iterators (each element
//! by reference or mutable reference) have no ordering guarantee in their
//! parallel implementations.
//!
//! `.axis_iter()` and `.axis_iter_mut()` also have parallel counterparts,
//! and their parallel iterators are indexed (and thus ordered) and exact length.
//!
//! (*) This regime of a custom trait instead of rayon’s own is since we
//! use this intermediate ndarray-parallel crate.
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


pub extern crate ndarray;
pub extern crate rayon;

/// Into- traits for creating parallelized iterators.
pub mod prelude {
    // happy and insane; ignorance is bluss
    pub use NdarrayIntoParallelIterator;
    pub use NdarrayIntoParallelRefIterator;
    pub use NdarrayIntoParallelRefMutIterator;

    #[doc(no_inline)]
    pub use rayon::prelude::{ParallelIterator, IndexedParallelIterator, ExactParallelIterator};

    pub use ext_traits::{
        ParApply1,
        ParApply2,
        ParApply3,
        ParApply4,
        ParApply5,
        ParApply6,
    };
}

pub use par::Parallel;
pub use into_traits::{
    NdarrayIntoParallelIterator,
    NdarrayIntoParallelRefIterator,
    NdarrayIntoParallelRefMutIterator,
};

mod par;
mod ext_traits;
mod into_traits;
mod into_impls;
