// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use std::mem::{forget, size_of};

use crate::imp_prelude::*;
use crate::{dimension, ArcArray1, ArcArray2};

/// Create an **[`Array`]** with one, two or
/// three dimensions.
///
/// ```
/// use ndarray::array;
/// let a1 = array![1, 2, 3, 4];
///
/// let a2 = array![[1, 2],
///                 [3, 4]];
///
/// let a3 = array![[[1, 2], [3, 4]],
///                 [[5, 6], [7, 8]]];
///
/// assert_eq!(a1.shape(), &[4]);
/// assert_eq!(a2.shape(), &[2, 2]);
/// assert_eq!(a3.shape(), &[2, 2, 2]);
/// ```
///
/// This macro uses `vec![]`, and has the same ownership semantics;
/// elements are moved into the resulting `Array`.
///
/// Use `array![...].into_shared()` to create an `ArcArray`.
#[macro_export]
macro_rules! array {
    ($([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {{
        $crate::Array3::from(vec![$([$([$($x,)*],)*],)*])
    }};
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {{
        $crate::Array2::from(vec![$([$($x,)*],)*])
    }};
    ($($x:expr),* $(,)*) => {{
        $crate::Array::from(vec![$($x,)*])
    }};
}

/// Create a zero-dimensional array with the element `x`.
pub fn arr0<A>(x: A) -> Array0<A> {
    unsafe { ArrayBase::from_shape_vec_unchecked((), vec![x]) }
}

/// Create a one-dimensional array with elements from `xs`.
pub fn arr1<A: Clone>(xs: &[A]) -> Array1<A> {
    ArrayBase::from(xs.to_vec())
}

/// Create a one-dimensional array with elements from `xs`.
pub fn rcarr1<A: Clone>(xs: &[A]) -> ArcArray1<A> {
    arr1(xs).into_shared()
}

/// Create a zero-dimensional array view borrowing `x`.
pub fn aview0<A>(x: &A) -> ArrayView0<'_, A> {
    unsafe { ArrayView::from_shape_ptr(Ix0(), x) }
}

/// Create a one-dimensional array view with elements borrowing `xs`.
///
/// ```
/// use ndarray::aview1;
///
/// let data = [1.0; 1024];
///
/// // Create a 2D array view from borrowed data
/// let a2d = aview1(&data).into_shape((32, 32)).unwrap();
///
/// assert_eq!(a2d.sum(), 1024.0);
/// ```
pub fn aview1<A>(xs: &[A]) -> ArrayView1<'_, A> {
    ArrayView::from(xs)
}

/// Create a two-dimensional array view with elements borrowing `xs`.
///
/// **Panics** if the product of non-zero axis lengths overflows `isize` (This can only occur if A
/// is zero-sized because slices cannot contain more than `isize::MAX` number of bytes).
pub fn aview2<A, const N: usize>(xs: &[[A; N]]) -> ArrayView2<'_, A> {
    ArrayView2::from(xs)
}

/// Create a one-dimensional read-write array view with elements borrowing `xs`.
///
/// ```
/// use ndarray::{aview_mut1, s};
/// // Create an array view over some data, then slice it and modify it.
/// let mut data = [0; 1024];
/// {
///     let mut a = aview_mut1(&mut data).into_shape((32, 32)).unwrap();
///     a.slice_mut(s![.., ..;3]).fill(5);
/// }
/// assert_eq!(&data[..10], [5, 0, 0, 5, 0, 0, 5, 0, 0, 5]);
/// ```
pub fn aview_mut1<A>(xs: &mut [A]) -> ArrayViewMut1<'_, A> {
    ArrayViewMut::from(xs)
}

/// Create a two-dimensional read-write array view with elements borrowing `xs`.
///
/// **Panics** if the product of non-zero axis lengths overflows `isize` (This can only occur if A
/// is zero-sized because slices cannot contain more than `isize::MAX` number of bytes).
///
/// # Example
///
/// ```
/// use ndarray::aview_mut2;
///
/// // The inner (nested) and outer arrays can be of any length.
/// let mut data = [[0.; 2]; 128];
/// {
///     // Make a 128 x 2 mut array view then turn it into 2 x 128
///     let mut a = aview_mut2(&mut data).reversed_axes();
///     // Make the first row ones and second row minus ones.
///     a.row_mut(0).fill(1.);
///     a.row_mut(1).fill(-1.);
/// }
/// // look at the start of the result
/// assert_eq!(&data[..3], [[1., -1.], [1., -1.], [1., -1.]]);
/// ```
pub fn aview_mut2<A, const N: usize>(xs: &mut [[A; N]]) -> ArrayViewMut2<'_, A> {
    ArrayViewMut2::from(xs)
}

/// Create a two-dimensional array with elements from `xs`.
///
/// ```
/// use ndarray::arr2;
///
/// let a = arr2(&[[1, 2, 3],
///                [4, 5, 6]]);
/// assert!(
///     a.shape() == [2, 3]
/// );
/// ```
pub fn arr2<A: Clone, const N: usize>(xs: &[[A; N]]) -> Array2<A> {
    Array2::from(xs.to_vec())
}

impl<A, const N: usize> From<Vec<[A; N]>> for Array2<A> {
    /// Converts the `Vec` of arrays to an owned 2-D array.
    ///
    /// **Panics** if the product of non-zero axis lengths overflows `isize`.
    fn from(mut xs: Vec<[A; N]>) -> Self {
        let dim = Ix2(xs.len(), N);
        let ptr = xs.as_mut_ptr();
        let cap = xs.capacity();
        let expand_len = dimension::size_of_shape_checked(&dim)
            .expect("Product of non-zero axis lengths must not overflow isize.");
        forget(xs);
        unsafe {
            let v = if size_of::<A>() == 0 {
                Vec::from_raw_parts(ptr as *mut A, expand_len, expand_len)
            } else if N == 0 {
                Vec::new()
            } else {
                // Guaranteed not to overflow in this case since A is non-ZST
                // and Vec never allocates more than isize bytes.
                let expand_cap = cap * N;
                Vec::from_raw_parts(ptr as *mut A, expand_len, expand_cap)
            };
            ArrayBase::from_shape_vec_unchecked(dim, v)
        }
    }
}

impl<A, const N: usize, const M: usize> From<Vec<[[A; M]; N]>> for Array3<A> {
    /// Converts the `Vec` of arrays to an owned 3-D array.
    ///
    /// **Panics** if the product of non-zero axis lengths overflows `isize`.
    fn from(mut xs: Vec<[[A; M]; N]>) -> Self {
        let dim = Ix3(xs.len(), N, M);
        let ptr = xs.as_mut_ptr();
        let cap = xs.capacity();
        let expand_len = dimension::size_of_shape_checked(&dim)
            .expect("Product of non-zero axis lengths must not overflow isize.");
        forget(xs);
        unsafe {
            let v = if size_of::<A>() == 0 {
                Vec::from_raw_parts(ptr as *mut A, expand_len, expand_len)
            } else if N == 0 || M == 0 {
                Vec::new()
            } else {
                // Guaranteed not to overflow in this case since A is non-ZST
                // and Vec never allocates more than isize bytes.
                let expand_cap = cap * N * M;
                Vec::from_raw_parts(ptr as *mut A, expand_len, expand_cap)
            };
            ArrayBase::from_shape_vec_unchecked(dim, v)
        }
    }
}

/// Create a two-dimensional array with elements from `xs`.
///
pub fn rcarr2<A: Clone, const N: usize>(xs: &[[A; N]]) -> ArcArray2<A> {
    arr2(xs).into_shared()
}

/// Create a three-dimensional array with elements from `xs`.
///
/// **Panics** if the slices are not all of the same length.
///
/// ```
/// use ndarray::arr3;
///
/// let a = arr3(&[[[1, 2],
///                 [3, 4]],
///                [[5, 6],
///                 [7, 8]],
///                [[9, 0],
///                 [1, 2]]]);
/// assert!(
///     a.shape() == [3, 2, 2]
/// );
/// ```
pub fn arr3<A: Clone, const N: usize, const M: usize>(xs: &[[[A; M]; N]]) -> Array3<A> {
    Array3::from(xs.to_vec())
}

/// Create a three-dimensional array with elements from `xs`.
pub fn rcarr3<A: Clone, const N: usize, const M: usize>(xs: &[[[A; M]; N]]) -> ArcArray<A, Ix3> {
    arr3(xs).into_shared()
}
