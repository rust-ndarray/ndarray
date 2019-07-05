// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::slice;

use crate::arraytraits::array_out_of_bounds;
use crate::dimension;
use crate::error::ShapeError;
use crate::imp_prelude::*;
use crate::{is_aligned, NdIndex, StrideShape};

use crate::{Baseiter, ElementsBase, ElementsBaseMut, Iter, IterMut};

use crate::iter::{self, AxisIter, AxisIterMut};

/// Methods for read-only array views.
impl<'a, A, D> ArrayView<'a, A, D>
where
    D: Dimension,
{
    /// Create a read-only array view borrowing its data from a slice.
    ///
    /// Checks whether `shape` are compatible with the slice's
    /// length, returning an `Err` if not compatible.
    ///
    /// ```
    /// use ndarray::ArrayView;
    /// use ndarray::arr3;
    /// use ndarray::ShapeBuilder;
    ///
    /// let s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    /// let a = ArrayView::from_shape((2, 3, 2).strides((1, 4, 2)),
    ///                               &s).unwrap();
    ///
    /// assert!(
    ///     a == arr3(&[[[0, 2],
    ///                  [4, 6],
    ///                  [8, 10]],
    ///                 [[1, 3],
    ///                  [5, 7],
    ///                  [9, 11]]])
    /// );
    /// assert!(a.strides() == &[1, 4, 2]);
    /// ```
    pub fn from_shape<Sh>(shape: Sh, xs: &'a [A]) -> Result<Self, ShapeError>
    where
        Sh: Into<StrideShape<D>>,
    {
        // eliminate the type parameter Sh as soon as possible
        Self::from_shape_impl(shape.into(), xs)
    }

    fn from_shape_impl(shape: StrideShape<D>, xs: &'a [A]) -> Result<Self, ShapeError> {
        let dim = shape.dim;
        let strides = shape.strides;
        if shape.custom {
            dimension::can_index_slice(xs, &dim, &strides)?;
        } else {
            dimension::can_index_slice_not_custom::<A, _>(xs, &dim)?;
        }
        unsafe { Ok(Self::new_(xs.as_ptr(), dim, strides)) }
    }

    /// Create an `ArrayView<A, D>` from shape information and a raw pointer to
    /// the elements.
    ///
    /// Unsafe because caller is responsible for ensuring all of the following:
    ///
    /// * The elements seen by moving `ptr` according to the shape and strides
    ///   must live at least as long as `'a` and must not be not mutably
    ///   aliased for the duration of `'a`.
    ///
    /// * `ptr` must be non-null and aligned, and it must be safe to
    ///   [`.offset()`] `ptr` by zero.
    ///
    /// * It must be safe to [`.offset()`] the pointer repeatedly along all
    ///   axes and calculate the `count`s for the `.offset()` calls without
    ///   overflow, even if the array is empty or the elements are zero-sized.
    ///
    ///   In other words,
    ///
    ///   * All possible pointers generated by moving along all axes must be in
    ///     bounds or one byte past the end of a single allocation with element
    ///     type `A`. The only exceptions are if the array is empty or the element
    ///     type is zero-sized. In these cases, `ptr` may be dangling, but it must
    ///     still be safe to [`.offset()`] the pointer along the axes.
    ///
    ///   * The offset in units of bytes between the least address and greatest
    ///     address by moving along all axes must not exceed `isize::MAX`. This
    ///     constraint prevents the computed offset, in bytes, from overflowing
    ///     `isize` regardless of the starting point due to past offsets.
    ///
    ///   * The offset in units of `A` between the least address and greatest
    ///     address by moving along all axes must not exceed `isize::MAX`. This
    ///     constraint prevents overflow when calculating the `count` parameter to
    ///     [`.offset()`] regardless of the starting point due to past offsets.
    ///
    /// * The product of non-zero axis lengths must not exceed `isize::MAX`.
    ///
    /// [`.offset()`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.offset
    pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *const A) -> Self
    where
        Sh: Into<StrideShape<D>>,
    {
        RawArrayView::from_shape_ptr(shape, ptr).deref_into_view()
    }

    /// Convert the view into an `ArrayView<'b, A, D>` where `'b` is a lifetime
    /// outlived by `'a'`.
    pub fn reborrow<'b>(self) -> ArrayView<'b, A, D>
    where
        'a: 'b,
    {
        unsafe { ArrayView::new_(self.as_ptr(), self.dim, self.strides) }
    }

    /// Split the array view along `axis` and return one view strictly before the
    /// split and one view after the split.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    ///
    /// Below, an illustration of `.split_at(Axis(2), 2)` on
    /// an array with shape 3 × 5 × 5.
    ///
    /// <img src="https://rust-ndarray.github.io/ndarray/images/split_at.svg" width="300px" height="271px">
    pub fn split_at(self, axis: Axis, index: Ix) -> (Self, Self) {
        unsafe {
            let (left, right) = self.into_raw_view().split_at(axis, index);
            (left.deref_into_view(), right.deref_into_view())
        }
    }

    /// Return the array’s data as a slice, if it is contiguous and in standard order.
    /// Return `None` otherwise.
    #[deprecated(note = "`into_slice` has been renamed to `to_slice`", since = "0.13.0")]
    pub fn into_slice(&self) -> Option<&'a [A]> {
        if self.is_standard_layout() {
            unsafe { Some(slice::from_raw_parts(self.ptr, self.len())) }
        } else {
            None
        }
    }

    /// Return the array’s data as a slice, if it is contiguous and in standard order.
    /// Return `None` otherwise.
    pub fn to_slice(&self) -> Option<&'a [A]> {
        if self.is_standard_layout() {
            unsafe { Some(slice::from_raw_parts(self.ptr, self.len())) }
        } else {
            None
        }
    }

    /// Converts to a raw array view.
    pub(crate) fn into_raw_view(self) -> RawArrayView<A, D> {
        unsafe { RawArrayView::new_(self.ptr, self.dim, self.strides) }
    }
}

/// Extra indexing methods for array views
///
/// These methods are very similar to regular indexing or calling of the
/// `get`/`get_mut` methods that we can use on any array or array view. The
/// difference here is in the length of lifetime in the resulting reference.
///
/// **Note** that the `ArrayView` (read-only) and `ArrayViewMut` (read-write) differ
/// in how they are allowed implement this trait -- `ArrayView`'s implementation
/// is usual. If you put in a `ArrayView<'a, T, D>` here, you get references
/// `&'a T` out.
///
/// For `ArrayViewMut` to obey the borrowing rules we have to consume the
/// view if we call any of these methods. (The equivalent of reborrow is
/// `.view_mut()` for read-write array views, but if you can use that,
/// then the regular indexing / `get_mut` should suffice, too.)
///
/// ```
/// use ndarray::IndexLonger;
/// use ndarray::ArrayView;
///
/// let data = [0.; 256];
/// let long_life_ref = {
///     // make a 16 × 16 array view
///     let view = ArrayView::from(&data[..]).into_shape((16, 16)).unwrap();
///
///     // index the view and with `IndexLonger`.
///     // Note here that we get a reference with a life that is derived from
///     // `data`, the base data, instead of being derived from the view
///     IndexLonger::index(&view, [0, 1])
/// };
///
/// // view goes out of scope
///
/// assert_eq!(long_life_ref, &0.);
///
/// ```
pub trait IndexLonger<I> {
    /// The type of the reference to the element that is produced, including
    /// its lifetime.
    type Output;
    /// Get a reference of a element through the view.
    ///
    /// This method is like `Index::index` but with a longer lifetime (matching
    /// the array view); which we can only do for the array view and not in the
    /// `Index` trait.
    ///
    /// See also [the `get` method][1] which works for all arrays and array
    /// views.
    ///
    /// [1]: struct.ArrayBase.html#method.get
    ///
    /// **Panics** if index is out of bounds.
    fn index(self, index: I) -> Self::Output;

    /// Get a reference of a element through the view.
    ///
    /// This method is like `ArrayBase::get` but with a longer lifetime (matching
    /// the array view); which we can only do for the array view and not in the
    /// `Index` trait.
    ///
    /// See also [the `get` method][1] (and [`get_mut`][2]) which works for all arrays and array
    /// views.
    ///
    /// [1]: struct.ArrayBase.html#method.get
    /// [2]: struct.ArrayBase.html#method.get_mut
    ///
    /// **Panics** if index is out of bounds.
    fn get(self, index: I) -> Option<Self::Output>;

    /// Get a reference of a element through the view without boundary check
    ///
    /// This method is like `elem` with a longer lifetime (matching the array
    /// view); which we can't do for general arrays.
    ///
    /// See also [the `uget` method][1] which works for all arrays and array
    /// views.
    ///
    /// [1]: struct.ArrayBase.html#method.uget
    ///
    /// **Note:** only unchecked for non-debug builds of ndarray.
    unsafe fn uget(self, index: I) -> Self::Output;
}

impl<'a, 'b, I, A, D> IndexLonger<I> for &'b ArrayView<'a, A, D>
where
    I: NdIndex<D>,
    D: Dimension,
{
    type Output = &'a A;

    /// Get a reference of a element through the view.
    ///
    /// This method is like `Index::index` but with a longer lifetime (matching
    /// the array view); which we can only do for the array view and not in the
    /// `Index` trait.
    ///
    /// See also [the `get` method][1] which works for all arrays and array
    /// views.
    ///
    /// [1]: struct.ArrayBase.html#method.get
    ///
    /// **Panics** if index is out of bounds.
    fn index(self, index: I) -> &'a A {
        debug_bounds_check!(self, index);
        unsafe { &*self.get_ptr(index).unwrap_or_else(|| array_out_of_bounds()) }
    }

    fn get(self, index: I) -> Option<&'a A> {
        unsafe { self.get_ptr(index).map(|ptr| &*ptr) }
    }

    /// Get a reference of a element through the view without boundary check
    ///
    /// This method is like `elem` with a longer lifetime (matching the array
    /// view); which we can't do for general arrays.
    ///
    /// See also [the `uget` method][1] which works for all arrays and array
    /// views.
    ///
    /// [1]: struct.ArrayBase.html#method.uget
    ///
    /// **Note:** only unchecked for non-debug builds of ndarray.
    unsafe fn uget(self, index: I) -> &'a A {
        debug_bounds_check!(self, index);
        &*self.as_ptr().offset(index.index_unchecked(&self.strides))
    }
}

/// Methods for read-write array views.
impl<'a, A, D> ArrayViewMut<'a, A, D>
where
    D: Dimension,
{
    /// Create a read-write array view borrowing its data from a slice.
    ///
    /// Checks whether `dim` and `strides` are compatible with the slice's
    /// length, returning an `Err` if not compatible.
    ///
    /// ```
    /// use ndarray::ArrayViewMut;
    /// use ndarray::arr3;
    /// use ndarray::ShapeBuilder;
    ///
    /// let mut s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    /// let mut a = ArrayViewMut::from_shape((2, 3, 2).strides((1, 4, 2)),
    ///                                      &mut s).unwrap();
    ///
    /// a[[0, 0, 0]] = 1;
    /// assert!(
    ///     a == arr3(&[[[1, 2],
    ///                  [4, 6],
    ///                  [8, 10]],
    ///                 [[1, 3],
    ///                  [5, 7],
    ///                  [9, 11]]])
    /// );
    /// assert!(a.strides() == &[1, 4, 2]);
    /// ```
    pub fn from_shape<Sh>(shape: Sh, xs: &'a mut [A]) -> Result<Self, ShapeError>
    where
        Sh: Into<StrideShape<D>>,
    {
        // eliminate the type parameter Sh as soon as possible
        Self::from_shape_impl(shape.into(), xs)
    }

    fn from_shape_impl(shape: StrideShape<D>, xs: &'a mut [A]) -> Result<Self, ShapeError> {
        let dim = shape.dim;
        let strides = shape.strides;
        if shape.custom {
            dimension::can_index_slice(xs, &dim, &strides)?;
        } else {
            dimension::can_index_slice_not_custom::<A, _>(xs, &dim)?;
        }
        unsafe { Ok(Self::new_(xs.as_mut_ptr(), dim, strides)) }
    }

    /// Create an `ArrayViewMut<A, D>` from shape information and a
    /// raw pointer to the elements.
    ///
    /// Unsafe because caller is responsible for ensuring all of the following:
    ///
    /// * The elements seen by moving `ptr` according to the shape and strides
    ///   must live at least as long as `'a` and must not be aliased for the
    ///   duration of `'a`.
    ///
    /// * `ptr` must be non-null and aligned, and it must be safe to
    ///   [`.offset()`] `ptr` by zero.
    ///
    /// * It must be safe to [`.offset()`] the pointer repeatedly along all
    ///   axes and calculate the `count`s for the `.offset()` calls without
    ///   overflow, even if the array is empty or the elements are zero-sized.
    ///
    ///   In other words,
    ///
    ///   * All possible pointers generated by moving along all axes must be in
    ///     bounds or one byte past the end of a single allocation with element
    ///     type `A`. The only exceptions are if the array is empty or the element
    ///     type is zero-sized. In these cases, `ptr` may be dangling, but it must
    ///     still be safe to [`.offset()`] the pointer along the axes.
    ///
    ///   * The offset in units of bytes between the least address and greatest
    ///     address by moving along all axes must not exceed `isize::MAX`. This
    ///     constraint prevents the computed offset, in bytes, from overflowing
    ///     `isize` regardless of the starting point due to past offsets.
    ///
    ///   * The offset in units of `A` between the least address and greatest
    ///     address by moving along all axes must not exceed `isize::MAX`. This
    ///     constraint prevents overflow when calculating the `count` parameter to
    ///     [`.offset()`] regardless of the starting point due to past offsets.
    ///
    /// * The product of non-zero axis lengths must not exceed `isize::MAX`.
    ///
    /// [`.offset()`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.offset
    pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *mut A) -> Self
    where
        Sh: Into<StrideShape<D>>,
    {
        RawArrayViewMut::from_shape_ptr(shape, ptr).deref_into_view_mut()
    }

    /// Convert the view into an `ArrayViewMut<'b, A, D>` where `'b` is a lifetime
    /// outlived by `'a'`.
    pub fn reborrow<'b>(mut self) -> ArrayViewMut<'b, A, D>
    where
        'a: 'b,
    {
        unsafe { ArrayViewMut::new_(self.as_mut_ptr(), self.dim, self.strides) }
    }

    /// Split the array view along `axis` and return one mutable view strictly
    /// before the split and one mutable view after the split.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    pub fn split_at(self, axis: Axis, index: Ix) -> (Self, Self) {
        unsafe {
            let (left, right) = self.into_raw_view_mut().split_at(axis, index);
            (left.deref_into_view_mut(), right.deref_into_view_mut())
        }
    }

    /// Return the array’s data as a slice, if it is contiguous and in standard order.
    /// Return `None` otherwise.
    pub fn into_slice(self) -> Option<&'a mut [A]> {
        self.into_slice_().ok()
    }
}

impl<'a, I, A, D> IndexLonger<I> for ArrayViewMut<'a, A, D>
where
    I: NdIndex<D>,
    D: Dimension,
{
    type Output = &'a mut A;

    /// Convert a mutable array view to a mutable reference of a element.
    ///
    /// This method is like `IndexMut::index_mut` but with a longer lifetime
    /// (matching the array view); which we can only do for the array view and
    /// not in the `Index` trait.
    ///
    /// See also [the `get_mut` method][1] which works for all arrays and array
    /// views.
    ///
    /// [1]: struct.ArrayBase.html#method.get_mut
    ///
    /// **Panics** if index is out of bounds.
    fn index(mut self, index: I) -> &'a mut A {
        debug_bounds_check!(self, index);
        unsafe {
            match self.get_ptr_mut(index) {
                Some(ptr) => &mut *ptr,
                None => array_out_of_bounds(),
            }
        }
    }

    /// Convert a mutable array view to a mutable reference of a element, with
    /// checked access.
    ///
    /// See also [the `get_mut` method][1] which works for all arrays and array
    /// views.
    ///
    /// [1]: struct.ArrayBase.html#method.get_mut
    ///
    fn get(mut self, index: I) -> Option<&'a mut A> {
        debug_bounds_check!(self, index);
        unsafe {
            match self.get_ptr_mut(index) {
                Some(ptr) => Some(&mut *ptr),
                None => None,
            }
        }
    }

    /// Convert a mutable array view to a mutable reference of a element without
    /// boundary check.
    ///
    /// See also [the `uget_mut` method][1] which works for all arrays and array
    /// views.
    ///
    /// [1]: struct.ArrayBase.html#method.uget_mut
    ///
    /// **Note:** only unchecked for non-debug builds of ndarray.
    unsafe fn uget(mut self, index: I) -> &'a mut A {
        debug_bounds_check!(self, index);
        &mut *self
            .as_mut_ptr()
            .offset(index.index_unchecked(&self.strides))
    }
}

/// Private array view methods
impl<'a, A, D> ArrayView<'a, A, D>
where
    D: Dimension,
{
    /// Create a new `ArrayView`
    ///
    /// Unsafe because: `ptr` must be valid for the given dimension and strides.
    #[inline(always)]
    pub(crate) unsafe fn new_(ptr: *const A, dim: D, strides: D) -> Self {
        ArrayView {
            data: ViewRepr::new(),
            ptr: ptr as *mut A,
            dim: dim,
            strides: strides,
        }
    }

    #[inline]
    pub(crate) fn into_base_iter(self) -> Baseiter<A, D> {
        unsafe { Baseiter::new(self.ptr, self.dim, self.strides) }
    }

    #[inline]
    pub(crate) fn into_elements_base(self) -> ElementsBase<'a, A, D> {
        ElementsBase::new(self)
    }

    pub(crate) fn into_iter_(self) -> Iter<'a, A, D> {
        Iter::new(self)
    }

    /// Return an outer iterator for this view.
    #[doc(hidden)] // not official
    #[deprecated(note = "This method will be replaced.")]
    pub fn into_outer_iter(self) -> iter::AxisIter<'a, A, D::Smaller>
    where
        D: RemoveAxis,
    {
        AxisIter::new(self, Axis(0))
    }
}

impl<'a, A, D> ArrayViewMut<'a, A, D>
where
    D: Dimension,
{
    /// Create a new `ArrayView`
    ///
    /// Unsafe because: `ptr` must be valid for the given dimension and strides.
    #[inline(always)]
    pub(crate) unsafe fn new_(ptr: *mut A, dim: D, strides: D) -> Self {
        if cfg!(debug_assertions) {
            assert!(!ptr.is_null(), "The pointer must be non-null.");
            assert!(is_aligned(ptr), "The pointer must be aligned.");
            dimension::max_abs_offset_check_overflow::<A, _>(&dim, &strides).unwrap();
        }
        ArrayViewMut {
            data: ViewRepr::new(),
            ptr: ptr,
            dim: dim,
            strides: strides,
        }
    }

    // Convert into a read-only view
    pub(crate) fn into_view(self) -> ArrayView<'a, A, D> {
        unsafe { ArrayView::new_(self.ptr, self.dim, self.strides) }
    }

    /// Converts to a mutable raw array view.
    pub(crate) fn into_raw_view_mut(self) -> RawArrayViewMut<A, D> {
        unsafe { RawArrayViewMut::new_(self.ptr, self.dim, self.strides) }
    }

    #[inline]
    pub(crate) fn into_base_iter(self) -> Baseiter<A, D> {
        unsafe { Baseiter::new(self.ptr, self.dim, self.strides) }
    }

    #[inline]
    pub(crate) fn into_elements_base(self) -> ElementsBaseMut<'a, A, D> {
        ElementsBaseMut::new(self)
    }

    pub(crate) fn into_slice_(self) -> Result<&'a mut [A], Self> {
        if self.is_standard_layout() {
            unsafe { Ok(slice::from_raw_parts_mut(self.ptr, self.len())) }
        } else {
            Err(self)
        }
    }

    pub(crate) fn into_iter_(self) -> IterMut<'a, A, D> {
        IterMut::new(self)
    }

    /// Return an outer iterator for this view.
    #[doc(hidden)] // not official
    #[deprecated(note = "This method will be replaced.")]
    pub fn into_outer_iter(self) -> iter::AxisIterMut<'a, A, D::Smaller>
    where
        D: RemoveAxis,
    {
        AxisIterMut::new(self, Axis(0))
    }
}
