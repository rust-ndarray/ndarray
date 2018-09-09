//! Various operations on `ndarray`s
//!
//! # Sorting
//! Sorting `ndarray`s can be achieved by using the
//! [`sort` functions from the standard library][https://doc.rust-lang.org/stable/std/primitive.slice.html#method.sort_unstable].
//! As a basic example, here is how to sort an array of integers using `sort_unstable`
//! from the standard library:
//! ```
//! use ndarray::*;
//!     let mut arr: Array1<i64> = Array::from_vec(vec![3, 2, 5, 1]);
//!     arr.as_slice_mut().unwrap().sort_unstable();
//!     println!("1: {:?}", arr);
//! ```
//! If we wish to sort floating point numbers, then we must specify how
//! to handle sorting `NaN`s or `inf`s.
//! ```
//! let mut arr: Array1<f64> = Array::from_vec(vec![3.0, 2.0, 5.0, 1.0]);
//! arr.as_slice_mut()
//!     .unwrap()
//!     .sort_unstable_by(|x, y| match x.partial_cmp(y) {
//!         Some(ord) => ord,
//!         None => panic!("Attempting to sort NaN's or Inf's"),
//!     });
//! ```
//! We can perform an argsort, that is, retrieving the indecies
//! that would sort the array.
//! ```
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
//! use rand::{thread_rng, Rng};
//! let mut arr: Array1<f64> = Array::from_vec(vec![3.0, 2.0, 5.0, 1.0]);
//! thread_rng().shuffle(arr.as_slice_mut().unwrap());
//! ```
//!
//! Shuffling two arrays can be done in unison, although at the cost of copying both arrays:
//! ```
//! let arr1: Array1<f64> = Array::from_vec(vec![3.0, -1.0, 8.0, 2.0]);
//! let arr2: Array1<f64> = Array::from_vec(vec![3.0, 2.0, 5.0, 1.0]);
//!
//! let mut indecies: Vec<usize> = (0..arr1.len_of(Axis(0))).collect();
//! thread_rng().shuffle(&mut indecies);
//! let arr1 = arr1.select(Axis(0), &indecies);
//! let arr2 = arr2.select(Axis(0), &indecies);
//! ```
