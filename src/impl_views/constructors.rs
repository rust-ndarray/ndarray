// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ptr::NonNull;

use crate::dimension;
use crate::dimension::offset_from_low_addr_ptr_to_logical_ptr;
use crate::error::ShapeError;
use crate::extension::nonnull::nonnull_debug_checked_from_ptr;
use crate::imp_prelude::*;
use crate::{is_aligned, StrideShape};

/// Methods for read-only array views.
impl<'a, A, D> ArrayView<'a, A, D>
where D: Dimension
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
    /// // advanced example where we are even specifying exact strides to use (which is optional).
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
    where Sh: Into<StrideShape<D>> {
        // eliminate the type parameter Sh as soon as possible
        Self::from_shape_impl(shape.into(), xs)
    }

    fn from_shape_impl(shape: StrideShape<D>, xs: &'a [A]) -> Result<Self, ShapeError> {
        let dim = shape.dim;
        dimension::can_index_slice_with_strides(xs, &dim, &shape.strides)?;
        let strides = shape.strides.strides_for_dim(&dim);
        unsafe {
            Ok(Self::new_(
                xs.as_ptr()
                    .add(offset_from_low_addr_ptr_to_logical_ptr(&dim, &strides)),
                dim,
                strides,
            ))
        }
    }

    /// Create an `ArrayView<A, D>` from shape information and a raw pointer to
    /// the elements.
    ///
    /// # Safety
    ///
    /// The caller is responsible for ensuring all of the following:
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
    /// * Strides must be non-negative.
    ///
    /// This function can use debug assertions to check some of these requirements,
    /// but it's not a complete check.
    ///
    /// [`.offset()`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.offset
    pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *const A) -> Self
    where Sh: Into<StrideShape<D>> {
        RawArrayView::from_shape_ptr(shape, ptr).deref_into_view()
    }
}

/// Methods for read-write array views.
impl<'a, A, D> ArrayViewMut<'a, A, D>
where D: Dimension
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
    where Sh: Into<StrideShape<D>> {
        // eliminate the type parameter Sh as soon as possible
        Self::from_shape_impl(shape.into(), xs)
    }

    fn from_shape_impl(shape: StrideShape<D>, xs: &'a mut [A]) -> Result<Self, ShapeError> {
        let dim = shape.dim;
        dimension::can_index_slice_with_strides(xs, &dim, &shape.strides)?;
        let strides = shape.strides.strides_for_dim(&dim);
        unsafe {
            Ok(Self::new_(
                xs.as_mut_ptr()
                    .add(offset_from_low_addr_ptr_to_logical_ptr(&dim, &strides)),
                dim,
                strides,
            ))
        }
    }

    /// Create an `ArrayViewMut<A, D>` from shape information and a
    /// raw pointer to the elements.
    ///
    /// # Safety
    ///
    /// The caller is responsible for ensuring all of the following:
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
    /// * Strides must be non-negative.
    ///
    /// This function can use debug assertions to check some of these requirements,
    /// but it's not a complete check.
    ///
    /// [`.offset()`]: https://doc.rust-lang.org/stable/std/primitive.pointer.html#method.offset
    pub unsafe fn from_shape_ptr<Sh>(shape: Sh, ptr: *mut A) -> Self
    where Sh: Into<StrideShape<D>> {
        RawArrayViewMut::from_shape_ptr(shape, ptr).deref_into_view_mut()
    }

    /// Convert the view into an `ArrayViewMut<'b, A, D>` where `'b` is a lifetime
    /// outlived by `'a'`.
    pub fn reborrow<'b>(self) -> ArrayViewMut<'b, A, D>
    where 'a: 'b {
        unsafe { ArrayViewMut::new(self.ptr, self.dim, self.strides) }
    }
}

/// Private array view methods
impl<'a, A, D> ArrayView<'a, A, D>
where D: Dimension
{
    /// Create a new `ArrayView`
    ///
    /// Unsafe because: `ptr` must be valid for the given dimension and strides.
    #[inline(always)]
    pub(crate) unsafe fn new(ptr: NonNull<A>, dim: D, strides: D) -> Self {
        if cfg!(debug_assertions) {
            assert!(is_aligned(ptr.as_ptr()), "The pointer must be aligned.");
            dimension::max_abs_offset_check_overflow::<A, _>(&dim, &strides).unwrap();
        }
        ArrayView::from_data_ptr(ViewRepr::new(), ptr).with_strides_dim(strides, dim)
    }

    /// Unsafe because: `ptr` must be valid for the given dimension and strides.
    #[inline]
    pub(crate) unsafe fn new_(ptr: *const A, dim: D, strides: D) -> Self {
        Self::new(nonnull_debug_checked_from_ptr(ptr as *mut A), dim, strides)
    }
}

impl<'a, A, D> ArrayViewMut<'a, A, D>
where D: Dimension
{
    /// Create a new `ArrayView`
    ///
    /// Unsafe because: `ptr` must be valid for the given dimension and strides.
    #[inline(always)]
    pub(crate) unsafe fn new(ptr: NonNull<A>, dim: D, strides: D) -> Self {
        if cfg!(debug_assertions) {
            assert!(is_aligned(ptr.as_ptr()), "The pointer must be aligned.");
            dimension::max_abs_offset_check_overflow::<A, _>(&dim, &strides).unwrap();
        }
        ArrayViewMut::from_data_ptr(ViewRepr::new(), ptr).with_strides_dim(strides, dim)
    }

    /// Create a new `ArrayView`
    ///
    /// Unsafe because: `ptr` must be valid for the given dimension and strides.
    #[inline(always)]
    pub(crate) unsafe fn new_(ptr: *mut A, dim: D, strides: D) -> Self {
        Self::new(nonnull_debug_checked_from_ptr(ptr), dim, strides)
    }
}
