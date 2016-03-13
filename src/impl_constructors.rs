// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Constructor methods for ndarray
//!
use libnum;

use imp_prelude::*;
use dimension;
use linspace;
use error::{self, ShapeError, ErrorKind};

/// Constructor methods for one-dimensional arrays.
///
/// Note that the constructor methods apply to `OwnedArray` and `RcArray`,
/// the two array types that have owned storage.
impl<S> ArrayBase<S, Ix>
    where S: DataOwned
{
    /// Create a one-dimensional array from a vector (no copying needed).
    ///
    /// ```rust
    /// use ndarray::OwnedArray;
    ///
    /// let array = OwnedArray::from_vec(vec![1., 2., 3., 4.]);
    /// ```
    pub fn from_vec(v: Vec<S::Elem>) -> ArrayBase<S, Ix> {
        unsafe { Self::from_vec_dim_unchecked(v.len() as Ix, v) }
    }

    /// Create a one-dimensional array from an iterable.
    ///
    /// ```rust
    /// use ndarray::{OwnedArray, arr1};
    ///
    /// let array = OwnedArray::from_iter((0..5).map(|x| x * x));
    /// assert!(array == arr1(&[0, 1, 4, 9, 16]))
    /// ```
    pub fn from_iter<I>(iterable: I) -> ArrayBase<S, Ix>
        where I: IntoIterator<Item=S::Elem>
    {
        Self::from_vec(iterable.into_iter().collect())
    }

    /// Create a one-dimensional array from the inclusive interval
    /// `[start, end]` with `n` elements. `F` must be a floating point type.
    ///
    /// ```rust
    /// use ndarray::{OwnedArray, arr1};
    ///
    /// let array = OwnedArray::linspace(0., 1., 5);
    /// assert!(array == arr1(&[0.0, 0.25, 0.5, 0.75, 1.0]))
    /// ```
    pub fn linspace<F>(start: F, end: F, n: usize) -> ArrayBase<S, Ix>
        where S: Data<Elem=F>,
              F: libnum::Float,
    {
        Self::from_vec(::iterators::to_vec(linspace::linspace(start, end, n)))
    }

    /// Create a one-dimensional array from the half-open interval
    /// `[start, end)` with elements spaced by `step`. `F` must be a floating point type.
    ///
    /// ```rust
    /// use ndarray::{OwnedArray, arr1};
    ///
    /// let array = OwnedArray::range(0., 5., 1.);
    /// assert!(array == arr1(&[0., 1., 2., 3., 4.]))
    /// ```
    pub fn range<F>(start: F, end: F, step: F) -> ArrayBase<S, Ix>
        where S: Data<Elem=F>,
              F: libnum::Float,
    {
        Self::from_vec(::iterators::to_vec(linspace::range(start, end, step)))
    }
}

/// Constructor methods for two-dimensional arrays.
impl<S, A> ArrayBase<S, (Ix, Ix)>
    where S: DataOwned<Elem=A>,
{
    /// Create an identity matrix of size `n` (square 2D array).
    ///
    /// **Panics** if `n * n` would overflow usize.
    pub fn eye(n: Ix) -> ArrayBase<S, (Ix, Ix)>
        where S: DataMut,
              A: Clone + libnum::Zero + libnum::One,
    {
        let mut eye = Self::zeros((n, n));
        for a_ii in eye.diag_mut() {
            *a_ii = A::one();
        }
        eye
    }
}

macro_rules! size_checked_unwrap {
    ($dim:expr) => {
        match $dim.size_checked() {
            Some(sz) => sz,
            None => panic!("ndarray: Shape too large, number of elements overflows usize"),
        }
    }
}

/// Constructor methods for n-dimensional arrays.
impl<S, A, D> ArrayBase<S, D>
    where S: DataOwned<Elem=A>,
          D: Dimension,
{
    /// Create an array with copies of `elem`, dimension `dim`.
    ///
    /// **Panics** if the number of elements in `dim` would overflow usize.
    ///
    /// ```
    /// use ndarray::OwnedArray;
    /// use ndarray::arr3;
    ///
    /// let a = OwnedArray::from_elem((2, 2, 2), 1.);
    ///
    /// assert!(
    ///     a == arr3(&[[[1., 1.],
    ///                  [1., 1.]],
    ///                 [[1., 1.],
    ///                  [1., 1.]]])
    /// );
    /// assert!(a.strides() == &[4, 2, 1]);
    /// ```
    pub fn from_elem(dim: D, elem: A) -> ArrayBase<S, D>
        where A: Clone
    {
        // Note: We don't need to check the case of a size between
        // isize::MAX -> usize::MAX; in this case, the vec constructor itself
        // panics.
        let size = size_checked_unwrap!(dim);
        let v = vec![elem; size];
        unsafe { Self::from_vec_dim_unchecked(dim, v) }
    }

    /// Create an array with copies of `elem`, dimension `dim` and fortran
    /// memory order.
    ///
    /// **Panics** if the number of elements would overflow usize.
    ///
    /// ```
    /// use ndarray::OwnedArray;
    ///
    /// let a = OwnedArray::from_elem_f((2, 2, 2), 1.);
    /// assert!(a.strides() == &[1, 2, 4]);
    /// ```
    pub fn from_elem_f(dim: D, elem: A) -> ArrayBase<S, D>
        where A: Clone
    {
        let size = size_checked_unwrap!(dim);
        let v = vec![elem; size];
        unsafe { Self::from_vec_dim_unchecked_f(dim, v) }
    }

    /// Create an array with zeros, dimension `dim`.
    ///
    /// **Panics** if the number of elements in `dim` would overflow usize.
    pub fn zeros(dim: D) -> ArrayBase<S, D>
        where A: Clone + libnum::Zero
    {
        Self::from_elem(dim, libnum::zero())
    }

    /// Create an array with zeros, dimension `dim` and fortran memory order.
    ///
    /// **Panics** if the number of elements in `dim` would overflow usize.
    pub fn zeros_f(dim: D) -> ArrayBase<S, D>
        where A: Clone + libnum::Zero
    {
        Self::from_elem_f(dim, libnum::zero())
    }

    /// Create an array with default values, dimension `dim`.
    ///
    /// **Panics** if the number of elements in `dim` would overflow usize.
    pub fn default(dim: D) -> ArrayBase<S, D>
        where A: Default
    {
        let v = (0..dim.size()).map(|_| A::default()).collect();
        unsafe { Self::from_vec_dim_unchecked(dim, v) }
    }

    /// Create an array from a vector (no copying needed).
    ///
    /// **Errors** if `dim` does not correspond to the number of elements in `v`.
    pub fn from_vec_dim(dim: D, v: Vec<A>) -> Result<ArrayBase<S, D>, ShapeError> {
        if dim.size_checked() != Some(v.len()) {
            return Err(error::incompatible_shapes(&v.len(), &dim));
        }
        unsafe { Ok(Self::from_vec_dim_unchecked(dim, v)) }
    }

    /// Create an array from a vector (no copying needed).
    ///
    /// Unsafe because dimension is unchecked, and must be correct.
    pub unsafe fn from_vec_dim_unchecked(dim: D, mut v: Vec<A>) -> ArrayBase<S, D> {
        debug_assert!(dim.size_checked() == Some(v.len()));
        ArrayBase {
            ptr: v.as_mut_ptr(),
            data: DataOwned::new(v),
            strides: dim.default_strides(),
            dim: dim,
        }
    }

    /// Create an array from a vector (with no copying needed),
    /// using fortran memory order to interpret the data.
    ///
    /// Unsafe because dimension is unchecked, and must be correct.
    pub unsafe fn from_vec_dim_unchecked_f(dim: D, mut v: Vec<A>) -> ArrayBase<S, D> {
        debug_assert!(dim.size_checked() == Some(v.len()));
        ArrayBase {
            ptr: v.as_mut_ptr(),
            data: DataOwned::new(v),
            strides: dim.fortran_strides(),
            dim: dim,
        }
    }

    /// Create an array from a vector and interpret it according to the
    /// provided dimensions and strides. No allocation needed.
    ///
    /// Checks whether `dim` and `strides` are compatible with the vector's
    /// length, returning an `Err` if not compatible.
    ///
    /// **Errors** if strides and dimensions can point out of bounds of `v`.<br>
    /// **Errors** if strides allow multiple indices to point to the same element.
    pub fn from_vec_dim_stride(dim: D, strides: D, v: Vec<A>)
        -> Result<ArrayBase<S, D>, ShapeError>
    {
        dimension::can_index_slice(&v, &dim, &strides).map(|_| {
            unsafe {
                Self::from_vec_dim_stride_unchecked(dim, strides, v)
            }
        })
    }

    /// Create an array from a vector and interpret it according to the
    /// provided dimensions and strides. No allocation needed.
    ///
    /// Unsafe because dimension and strides are unchecked.
    pub unsafe fn from_vec_dim_stride_unchecked(dim: D, strides: D, mut v: Vec<A>)
        -> ArrayBase<S, D>
    {
        // debug check for issues that indicates wrong use of this constructor
        debug_assert!(match dimension::can_index_slice(&v, &dim, &strides) {
            Ok(_) => true,
            Err(ref e) => match e.kind() {
                ErrorKind::OutOfBounds => false,
                ErrorKind::RangeLimited => false,
                _ => true,
            }
        });
        ArrayBase {
            ptr: v.as_mut_ptr(),
            data: DataOwned::new(v),
            strides: strides,
            dim: dim
        }
    }

}
