// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Constructor methods for ndarray
//!
//!

use libnum::{Zero, One, Float};

use imp_prelude::*;
use {Shape, StrideShape};
use dimension;
use linspace;
use error::{self, ShapeError, ErrorKind};
use Indexes;
use iterators::to_vec;

/// Constructor methods for one-dimensional arrays.
///
/// Note that the constructor methods apply to `Array` and `RcArray`,
/// the two array types that have owned storage.
impl<S> ArrayBase<S, Ix>
    where S: DataOwned
{
    /// Create a one-dimensional array from a vector (no copying needed).
    ///
    /// ```rust
    /// use ndarray::Array;
    ///
    /// let array = Array::from_vec(vec![1., 2., 3., 4.]);
    /// ```
    pub fn from_vec(v: Vec<S::Elem>) -> ArrayBase<S, Ix> {
        unsafe { Self::from_shape_vec_unchecked(v.len() as Ix, v) }
    }

    /// Create a one-dimensional array from an iterable.
    ///
    /// ```rust
    /// use ndarray::{Array, arr1};
    ///
    /// let array = Array::from_iter((0..5).map(|x| x * x));
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
    /// use ndarray::{Array, arr1};
    ///
    /// let array = Array::linspace(0., 1., 5);
    /// assert!(array == arr1(&[0.0, 0.25, 0.5, 0.75, 1.0]))
    /// ```
    pub fn linspace<F>(start: F, end: F, n: usize) -> ArrayBase<S, Ix>
        where S: Data<Elem=F>,
              F: Float,
    {
        Self::from_vec(to_vec(linspace::linspace(start, end, n)))
    }

    /// Create a one-dimensional array from the half-open interval
    /// `[start, end)` with elements spaced by `step`. `F` must be a floating point type.
    ///
    /// ```rust
    /// use ndarray::{Array, arr1};
    ///
    /// let array = Array::range(0., 5., 1.);
    /// assert!(array == arr1(&[0., 1., 2., 3., 4.]))
    /// ```
    pub fn range<F>(start: F, end: F, step: F) -> ArrayBase<S, Ix>
        where S: Data<Elem=F>,
              F: Float,
    {
        Self::from_vec(to_vec(linspace::range(start, end, step)))
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
              A: Clone + Zero + One,
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
    /// Create an array with copies of `elem`, shape `shape`.
    ///
    /// **Panics** if the number of elements in `shape` would overflow usize.
    ///
    /// ```
    /// use ndarray::{Array, arr3, ShapeBuilder};
    ///
    /// let a = Array::from_elem((2, 2, 2), 1.);
    ///
    /// assert!(
    ///     a == arr3(&[[[1., 1.],
    ///                  [1., 1.]],
    ///                 [[1., 1.],
    ///                  [1., 1.]]])
    /// );
    /// assert!(a.strides() == &[4, 2, 1]);
    ///
    /// let b = Array::from_elem((2, 2, 2).f(), 1.);
    /// assert!(b.strides() == &[1, 2, 4]);
    /// ```
    pub fn from_elem<Sh>(shape: Sh, elem: A) -> ArrayBase<S, D>
        where A: Clone,
              Sh: Into<Shape<D>>,
    {
        // Note: We don't need to check the case of a size between
        // isize::MAX -> usize::MAX; in this case, the vec constructor itself
        // panics.
        let shape = shape.into();
        let size = size_checked_unwrap!(shape.dim);
        let v = vec![elem; size];
        unsafe { Self::from_shape_vec_unchecked(shape, v) }
    }

    /// Create an array with zeros, shape `shape`.
    ///
    /// **Panics** if the number of elements in `shape` would overflow usize.
    pub fn zeros<Sh>(shape: Sh) -> ArrayBase<S, D>
        where A: Clone + Zero,
              Sh: Into<Shape<D>>,
    {
        Self::from_elem(shape, A::zero())
    }

    /// Create an array with default values, shape `shape`
    ///
    /// **Panics** if the number of elements in `shape` would overflow usize.
    pub fn default<Sh>(shape: Sh) -> ArrayBase<S, D>
        where A: Default,
              Sh: Into<Shape<D>>,
    {
        let shape = shape.into();
        let v = to_vec((0..shape.dim.size()).map(|_| A::default()));
        unsafe { Self::from_shape_vec_unchecked(shape, v) }
    }

    /// Create an array with values created by the function `f`.
    ///
    /// The elements are visited in arbitirary order.
    ///
    /// **Panics** if the number of elements in `shape` would overflow usize.
    pub fn from_shape_fn<Sh, F>(shape: Sh, f: F) -> ArrayBase<S, D>
        where Sh: Into<Shape<D>>,
              F: FnMut(D) -> A,
    {
        let shape = shape.into();
        let v = to_vec(Indexes::new(shape.dim.clone()).map(f));
        unsafe { Self::from_shape_vec_unchecked(shape, v) }
    }

    /// Create an array with the given shape from a vector. (No cloning of
    /// elements needed.)
    ///
    /// ---- 
    ///
    /// For a contiguous c- or f-order shape, the following applies:
    ///
    /// **Errors** if `shape` does not correspond to the number of elements in `v`.
    ///
    /// ---- 
    ///
    /// For custom strides, the following applies:
    ///
    /// **Errors** if strides and dimensions can point out of bounds of `v`.<br>
    /// **Errors** if strides allow multiple indices to point to the same element.
    ///
    /// ```
    /// use ndarray::prelude::*;
    ///
    /// let a = Array::from_shape_vec((2, 2), vec![1., 2., 3., 4.]);
    /// assert!(a.is_ok());
    ///
    /// let b = Array::from_shape_vec((2, 2).strides((1, 2)),
    ///                                    vec![1., 2., 3., 4.]).unwrap();
    /// assert!(
    ///     b == arr2(&[[1., 3.],
    ///                 [2., 4.]])
    /// );
    /// ```
    pub fn from_shape_vec<Sh>(shape: Sh, v: Vec<A>) -> Result<ArrayBase<S, D>, ShapeError>
        where Sh: Into<StrideShape<D>>,
    {
        // eliminate the type parameter Sh as soon as possible
        Self::from_shape_vec_impl(shape.into(), v)
    }

    fn from_shape_vec_impl(shape: StrideShape<D>, v: Vec<A>) -> Result<ArrayBase<S, D>, ShapeError>
    {
        if shape.custom {
            Self::from_vec_dim_stride(shape.dim, shape.strides, v)
        } else {
            let dim = shape.dim;
            let strides = shape.strides;
            if dim.size_checked() != Some(v.len()) {
                return Err(error::incompatible_shapes(&v.len(), &dim));
            }
            unsafe { Ok(Self::from_vec_dim_stride_unchecked(dim, strides, v)) }
        }
    }

    /// Create an array from a vector and interpret it according to the
    /// provided dimensions and strides. (No cloning of elements needed.)
    ///
    /// Unsafe because dimension and strides are unchecked.
    pub unsafe fn from_shape_vec_unchecked<Sh>(shape: Sh, v: Vec<A>) -> ArrayBase<S, D>
        where Sh: Into<StrideShape<D>>,
    {
        let shape = shape.into();
        Self::from_vec_dim_stride_unchecked(shape.dim, shape.strides, v)
    }

    fn from_vec_dim_stride(dim: D, strides: D, v: Vec<A>)
        -> Result<ArrayBase<S, D>, ShapeError>
    {
        dimension::can_index_slice(&v, &dim, &strides).map(|_| {
            unsafe {
                Self::from_vec_dim_stride_unchecked(dim, strides, v)
            }
        })
    }

    unsafe fn from_vec_dim_stride_unchecked(dim: D, strides: D, mut v: Vec<A>)
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
