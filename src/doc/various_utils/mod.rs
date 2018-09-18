//! Various operations on `ndarray`s.
//!
//! # Sorting
//! Sorting `ndarray`s can be achieved by using the
//! [`sort` functions from the standard library][https://doc.rust-lang.org/stable/std/primitive.slice.html#method.sort_unstable].
//! As a basic example, here is how to sort an array of integers using `sort_unstable`
//! from the standard library:
//! ```
//! extern crate ndarray;
//! use ndarray::*;
//! let mut arr: Array1<i64> = Array::from_vec(vec![3, 2, 5, 1]);
//! arr.as_slice_mut().unwrap().sort_unstable();
//! ```
//!
//! If we wish to sort floating point numbers, then we must specify how
//! to handle sorting `NaN`s or `inf`s.
//! ```
//! use ndarray::*;
//! let mut arr: Array1<f64> = Array::from_vec(vec![3.0, 2.0, 5.0, 1.0]);
//! arr.as_slice_mut()
//!     .unwrap()
//!     .sort_unstable_by(|x, y| match x.partial_cmp(y) {
//!         Some(ord) => ord,
//!         None => panic!("Attempting to sort NaN's or Inf's"),
//!     });
//! ```
//!
//! We can perform an argsort, that is, retrieving the indices
//! that would sort the array.
//! ```
//! use ndarray::*;
//! fn argsort<D: Dimension, E: Ord>(arr: &Array<E, D>) -> Array1<usize> {
//!     let mut zipped: Array1<(usize, &E)> = arr.into_iter().enumerate().collect();
//!     zipped
//!         .as_slice_mut()
//!         .unwrap()
//!         .sort_unstable_by_key(|&(_, val)| val);
//!     zipped.map(|(idx, _)| *idx)
//! }
//! ```
//!
//! # Shuffling
//! Similarly, shuffling can be performed using the [Rand Crate][https://crates.io/crates/rand]
//! ```
//! extern crate rand;
//! extern crate ndarray;
//! use ndarray::*;
//! use rand::{thread_rng, Rng};
//! let mut arr: Array1<f64> = Array::from_vec(vec![3.0, 2.0, 5.0, 1.0]);
//! thread_rng().shuffle(arr.as_slice_mut().unwrap());
//! ```
//!
//! Two arrays can be shuffled in unison, at the cost of copying both arrays:
//! ```
//! extern crate rand;
//! extern crate ndarray;
//! use ndarray::*;
//! use rand::{thread_rng, Rng};
//! let arr1: Array1<f64> = Array::from_vec(vec![3.0, -1.0, 8.0, 2.0]);
//! let arr2: Array1<f64> = Array::from_vec(vec![3.0, 2.0, 5.0, 1.0]);
//!
//! let mut indices: Vec<usize> = (0..arr1.len_of(Axis(0))).collect();
//! thread_rng().shuffle(&mut indices);
//! let arr1 = arr1.select(Axis(0), &indices);
//! let arr2 = arr2.select(Axis(0), &indices);
//! ```
//!
//! # On mutating non-contiguous vs. contiguous arrays
//! Approaches using [`.as_slice_mut()`][https://docs.rs/ndarray/0.12.0/ndarray/struct.ArrayBase.html#method.as_slice_mut] only work if
//! the array/view is contiguous. This means that, for example, trying to sort a column in a 2-D row-major array doesn't work with this approach.
//!```
//!use ndarray::prelude::*;
//!
//!fn main() {
//!    let mut a = arr2(&[[1, 2], [3, 4]]);
//!    // a.column_mut(0).as_slice_mut().unwrap().sort_unstable();
//!}
//!```
//!This panics with "called `Option::unwrap()` on a `None` value" since the column is non-contiguous.
//!
//! There is not yet a good way to sort/shuffle non-contiguous arrays/views. If the array/view is non-contiguous,
//! there are basically three options at the moment:
//! * Copy the data into a continuous Vec/array/view, and then sort/shuffle the copy. This doesn't modify the order of the original data.
//! * Sort/shuffle a slice of indices with `.sort_by_key()`/`.shuffle()`, and then rearrange the data according to the sorted/shuffled indices. [][examples/sort-axis.rs] provides an example of this.
