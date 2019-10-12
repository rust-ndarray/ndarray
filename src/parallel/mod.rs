//! Parallelization features for ndarray.
//!
//! Parallelization features are based on the crate [rayon] and its parallel
//! iterators. Ndarray implements the parallel iterable traits for arrays
//! and array views, for some of its iterators and for [Zip].
//! There are also directly parallelized methods on arrays and on [Zip].
//!
//! This requires the crate feature `rayon` to be enabled.
//!
//! The following types implement parallel iterators, accessed using these
//! methods:
//!
//! - [`Array`], [`ArcArray`]: `.par_iter()` and `.par_iter_mut()`
//! - [`ArrayView`](ArrayView): `.into_par_iter()`
//! - [`ArrayViewMut`](ArrayViewMut): `.into_par_iter()`
//! - [`AxisIter`](iter::AxisIter), [`AxisIterMut`](iter::AxisIterMut): `.into_par_iter()`
//! - [`AxisChunksIter`](iter::AxisChunksIter), [`AxisChunksIterMut`](iter::AxisChunksIterMut): `.into_par_iter()`
//! - [`Zip`] `.into_par_iter()`
//!
//! The following other parallelized methods exist:
//!
//! - [`ArrayBase::par_map_inplace()`]
//! - [`ArrayBase::par_mapv_inplace()`]
//! - [`Zip::par_apply()`] (all arities)
//!
//! Note that you can use the parallel iterator for [Zip] to access all other
//! rayon parallel iterator methods.
//!
//! Only the axis iterators are indexed parallel iterators, the rest are all
//! “unindexed”. Use ndarray’s [Zip] for lock step parallel iteration of
//! multiple arrays or producers at a time.
//!
//! # Examples
//!
//! ## Arrays and array views
//!
//! Compute the exponential of each element in an array, parallelized.
//!
//! ```
//! extern crate ndarray;
//!
//! use ndarray::Array2;
//! use ndarray::parallel::prelude::*;
//!
//! fn main() {
//!     let mut a = Array2::<f64>::zeros((128, 128));
//!
//!     // Parallel versions of regular array methods
//!     a.par_map_inplace(|x| *x = x.exp());
//!     a.par_mapv_inplace(f64::exp);
//!
//!     // You can also use the parallel iterator directly
//!     a.par_iter_mut().for_each(|x| *x = x.exp());
//! }
//! ```
//!
//! ## Axis iterators
//!
//! Use the parallel `.axis_iter()` to compute the sum of each row.
//!
//! ```
//! extern crate ndarray;
//!
//! use ndarray::Array;
//! use ndarray::Axis;
//! use ndarray::parallel::prelude::*;
//!
//! fn main() {
//!     let a = Array::linspace(0., 63., 64).into_shape((4, 16)).unwrap();
//!     let mut sums = Vec::new();
//!     a.axis_iter(Axis(0))
//!      .into_par_iter()
//!      .map(|row| row.sum())
//!      .collect_into_vec(&mut sums);
//!
//!     assert_eq!(sums, [120., 376., 632., 888.]);
//! }
//! ```
//!
//! ## Axis chunks iterators
//!
//! Use the parallel `.axis_chunks_iter()` to process your data in chunks.
//!
//! ```
//! extern crate ndarray;
//!
//! use ndarray::Array;
//! use ndarray::Axis;
//! use ndarray::parallel::prelude::*;
//!
//! fn main() {
//!     let a = Array::linspace(0., 63., 64).into_shape((4, 16)).unwrap();
//!     let mut shapes = Vec::new();
//!     a.axis_chunks_iter(Axis(0), 3)
//!      .into_par_iter()
//!      .map(|chunk| chunk.shape().to_owned())
//!      .collect_into_vec(&mut shapes);
//!
//!     assert_eq!(shapes, [vec![3, 16], vec![1, 16]]);
//! }
//! ```
//!
//! ## Zip
//!
//! Use zip for lock step function application across several arrays
//!
//! ```
//! extern crate ndarray;
//!
//! use ndarray::Array3;
//! use ndarray::Zip;
//!
//! type Array3f64 = Array3<f64>;
//!
//! fn main() {
//!     const N: usize = 128;
//!     let a = Array3f64::from_elem((N, N, N), 1.);
//!     let b = Array3f64::from_elem(a.dim(), 2.);
//!     let mut c = Array3f64::zeros(a.dim());
//!
//!     Zip::from(&mut c)
//!         .and(&a)
//!         .and(&b)
//!         .par_apply(|c, &a, &b| {
//!             *c += a - b;
//!         });
//! }
//! ```

/// Into- traits for creating parallelized iterators and/or using [`par_azip!`]
pub mod prelude {
    #[doc(no_inline)]
    pub use rayon::prelude::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    };

    pub use super::par_azip;
}

pub use self::par::Parallel;
pub use crate::par_azip;

mod impl_par_methods;
mod into_impls;
mod par;
mod zipmacro;
