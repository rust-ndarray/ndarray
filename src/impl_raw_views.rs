use dimension;
use imp_prelude::*;
use {is_aligned, StrideShape};

impl<A, D> RawArrayView<A, D>
where
    D: Dimension,
{
    /// Create a new `RawArrayView`.
    ///
    /// Unsafe because caller is responsible for ensuring that the array will
    /// meet all of the invariants of the `ArrayBase` type.
    #[inline(always)]
    pub(crate) unsafe fn new_(ptr: *const A, dim: D, strides: D) -> Self {
        RawArrayView {
            data: RawViewRepr::new(),
            ptr: ptr as *mut A,
            dim: dim,
            strides: strides,
        }
    }

    /// Create an `RawArrayView<A, D>` from shape information and a raw pointer
    /// to the elements.
    ///
    /// Unsafe because caller is responsible for ensuring all of the following:
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
        let shape = shape.into();
        let dim = shape.dim;
        let strides = shape.strides;
        if cfg!(debug_assertions) {
            assert!(!ptr.is_null(), "The pointer must be non-null.");
            assert!(is_aligned(ptr), "The pointer must be aligned.");
            dimension::max_abs_offset_check_overflow::<A, _>(&dim, &strides).unwrap();
        }
        RawArrayView::new_(ptr, dim, strides)
    }

    /// Return a read-only view of the array.
    ///
    /// **Warning** from a safety standpoint, this is equivalent to
    /// dereferencing a raw pointer for every element in the array. You must
    /// ensure that all of the data is valid and choose the correct lifetime.
    #[inline]
    pub unsafe fn deref_view<'a>(&self) -> ArrayView<'a, A, D> {
        ArrayView::new_(self.ptr, self.dim.clone(), self.strides.clone())
    }

    /// Converts to a read-only view of the array.
    ///
    /// **Warning** from a safety standpoint, this is equivalent to
    /// dereferencing a raw pointer for every element in the array. You must
    /// ensure that all of the data is valid and choose the correct lifetime.
    #[inline]
    pub unsafe fn deref_into_view<'a>(self) -> ArrayView<'a, A, D> {
        ArrayView::new_(self.ptr, self.dim, self.strides)
    }
}

impl<A, D> RawArrayViewMut<A, D>
where
    D: Dimension,
{
    /// Create a new `RawArrayViewMut`.
    ///
    /// Unsafe because caller is responsible for ensuring that the array will
    /// meet all of the invariants of the `ArrayBase` type.
    #[inline(always)]
    pub(crate) unsafe fn new_(ptr: *mut A, dim: D, strides: D) -> Self {
        RawArrayViewMut {
            data: RawViewRepr::new(),
            ptr: ptr,
            dim: dim,
            strides: strides,
        }
    }

    /// Create an `RawArrayViewMut<A, D>` from shape information and a raw
    /// pointer to the elements.
    ///
    /// Unsafe because caller is responsible for ensuring all of the following:
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
        let shape = shape.into();
        let dim = shape.dim;
        let strides = shape.strides;
        if cfg!(debug_assertions) {
            assert!(!ptr.is_null(), "The pointer must be non-null.");
            assert!(is_aligned(ptr), "The pointer must be aligned.");
            dimension::max_abs_offset_check_overflow::<A, _>(&dim, &strides).unwrap();
        }
        RawArrayViewMut::new_(ptr, dim, strides)
    }

    /// Return a read-only view of the array
    ///
    /// **Warning** from a safety standpoint, this is equivalent to
    /// dereferencing a raw pointer for every element in the array. You must
    /// ensure that all of the data is valid and choose the correct lifetime.
    #[inline]
    pub unsafe fn deref_view<'a>(&self) -> ArrayView<'a, A, D> {
        ArrayView::new_(self.ptr, self.dim.clone(), self.strides.clone())
    }

    /// Return a read-write view of the array
    ///
    /// **Warning** from a safety standpoint, this is equivalent to
    /// dereferencing a raw pointer for every element in the array. You must
    /// ensure that all of the data is valid and choose the correct lifetime.
    #[inline]
    pub unsafe fn deref_view_mut<'a>(&mut self) -> ArrayViewMut<'a, A, D> {
        ArrayViewMut::new_(self.ptr, self.dim.clone(), self.strides.clone())
    }

    /// Converts to a read-only view of the array.
    ///
    /// **Warning** from a safety standpoint, this is equivalent to
    /// dereferencing a raw pointer for every element in the array. You must
    /// ensure that all of the data is valid and choose the correct lifetime.
    #[inline]
    pub unsafe fn deref_into_view<'a>(self) -> ArrayView<'a, A, D> {
        ArrayView::new_(self.ptr, self.dim, self.strides)
    }

    /// Converts to a mutable view of the array.
    ///
    /// **Warning** from a safety standpoint, this is equivalent to
    /// dereferencing a raw pointer for every element in the array. You must
    /// ensure that all of the data is valid and choose the correct lifetime.
    #[inline]
    pub unsafe fn deref_into_view_mut<'a>(self) -> ArrayViewMut<'a, A, D> {
        ArrayViewMut::new_(self.ptr, self.dim, self.strides)
    }
}