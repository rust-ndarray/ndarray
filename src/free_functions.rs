// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::slice;

use imp_prelude::*;

/// Create a zero-dimensional array with the element `x`.
pub fn arr0<A>(x: A) -> Array0<A>
{
    unsafe { ArrayBase::from_shape_vec_unchecked((), vec![x]) }
}

/// Create a one-dimensional array with elements from `xs`.
pub fn arr1<A: Clone>(xs: &[A]) -> Array1<A> {
    ArrayBase::from_vec(xs.to_vec())
}

/// Create a one-dimensional array with elements from `xs`.
pub fn rcarr1<A: Clone>(xs: &[A]) -> RcArray<A, Ix1> {
    arr1(xs).into_shared()
}

/// Create a zero-dimensional array view borrowing `x`.
pub fn aview0<A>(x: &A) -> ArrayView0<A> {
    unsafe { ArrayView::new_(x, Ix0(), Ix0()) }
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
/// assert!(
///     a2d.scalar_sum() == 1024.0
/// );
/// ```
pub fn aview1<A>(xs: &[A]) -> ArrayView1<A> {
    ArrayView::from(xs)
}

/// Create a two-dimensional array view with elements borrowing `xs`.
pub fn aview2<A, V: FixedInitializer<Elem=A>>(xs: &[V]) -> ArrayView2<A> {
    let cols = V::len();
    let rows = xs.len();
    let data = unsafe {
        slice::from_raw_parts(xs.as_ptr() as *const A, cols * rows)
    };
    let dim = [rows as Ix, cols as Ix];
    unsafe {
        let strides = dim.default_strides();
        ArrayView::new_(data.as_ptr(), dim, strides)
    }
}

/// Create a one-dimensional read-write array view with elements borrowing `xs`.
///
/// ```
/// #[macro_use(s)]
/// extern crate ndarray;
///
/// use ndarray::aview_mut1;
///
/// // Create an array view over some data, then slice it and modify it.
/// fn main() {
///     let mut data = [0; 1024];
///     {
///         let mut a = aview_mut1(&mut data).into_shape((32, 32)).unwrap();
///         a.slice_mut(s![.., ..;3]).assign_scalar(&5);
///     }
///     assert_eq!(&data[..10], [5, 0, 0, 5, 0, 0, 5, 0, 0, 5]);
/// }
/// ```
pub fn aview_mut1<A>(xs: &mut [A]) -> ArrayViewMut1<A> {
    ArrayViewMut::from(xs)
}

/// Fixed-size array used for array initialization
pub unsafe trait FixedInitializer {
    type Elem;
    fn as_init_slice(&self) -> &[Self::Elem];
    fn len() -> usize;
}

macro_rules! impl_arr_init {
    (__impl $n: expr) => (
        unsafe impl<T> FixedInitializer for [T;  $n] {
            type Elem = T;
            fn as_init_slice(&self) -> &[T] { self }
            fn len() -> usize { $n }
        }
    );
    () => ();
    ($n: expr, $($m:expr,)*) => (
        impl_arr_init!(__impl $n);
        impl_arr_init!($($m,)*);
    )

}

impl_arr_init!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,);

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
pub fn arr2<A: Clone, V: FixedInitializer<Elem = A>>(xs: &[V]) -> Array2<A> {
    let (m, n) = (xs.len() as Ix, V::len() as Ix);
    let dim = [m, n];
    let mut result = Vec::<A>::with_capacity(dim.size());
    for snd in xs {
        result.extend_from_slice(snd.as_init_slice());
    }
    unsafe {
        ArrayBase::from_shape_vec_unchecked(dim, result)
    }
}

/// Create a two-dimensional array with elements from `xs`.
///
pub fn rcarr2<A: Clone, V: FixedInitializer<Elem = A>>(xs: &[V]) -> RcArray<A, Ix2> {
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
pub fn arr3<A: Clone, V: FixedInitializer<Elem=U>, U: FixedInitializer<Elem=A>>(xs: &[V])
    -> Array3<A>
{
    let dim = (xs.len() as Ix, V::len() as Ix, U::len() as Ix);
    let mut result = Vec::<A>::with_capacity(dim.size());
    for snd in xs {
        for thr in snd.as_init_slice() {
            result.extend_from_slice(thr.as_init_slice());
        }
    }
    unsafe {
        ArrayBase::from_shape_vec_unchecked(dim, result)
    }
}

/// Create a three-dimensional array with elements from `xs`.
pub fn rcarr3<A: Clone, V: FixedInitializer<Elem=U>, U: FixedInitializer<Elem=A>>(xs: &[V])
    -> RcArray<A, Ix3>
{
    arr3(xs).into_shared()
}
