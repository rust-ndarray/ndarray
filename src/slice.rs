// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use std::ops::{Deref, Range, RangeFrom, RangeFull, RangeTo};
use std::fmt;
use std::marker::PhantomData;
use super::{Dimension, Ixs};

/// A slice (range with step size).
///
/// Negative `begin` or `end` indexes are counted from the back of the axis. If
/// `end` is `None`, the slice extends to the end of the axis.
///
/// ## Examples
///
/// `Slice::new(0, None, 1)` is the full range of an axis. It can also be
/// created with `Slice::from(..)`. The Python equivalent is `[:]`.
///
/// `Slice::new(a, b, 2)` is every second element from `a` until `b`. It can
/// also be created with `Slice::from(a..b).step_by(2)`. The Python equivalent
/// is `[a:b:2]`.
///
/// `Slice::new(a, None, -1)` is every element, from `a` until the end, in
/// reverse order. It can also be created with `Slice::from(a..).step_by(-1)`.
/// The Python equivalent is `[a::-1]`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Slice {
    pub start: isize,
    pub end: Option<isize>,
    pub step: isize,
}

impl Slice {
    pub fn new(start: isize, end: Option<isize>, step: isize) -> Slice {
        Slice {
            start,
            end,
            step,
        }
    }

    /// Returns a new `Slice` with the given step size.
    #[inline]
    pub fn step_by(self, step: Ixs) -> Self {
        Slice { step, ..self }
    }

    /// Returns the `n`th index in this slice.
    ///
    /// The `n` argument starts from zero, so `nth(0)` is the first index. If
    /// `n` is greater than or equal to the length of the slice, then the
    /// return value is `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::Slice;
    ///
    /// // indices 0, 1, 2, 3, 4
    /// assert_eq!(Slice::new(0, Some(5), 1).nth(2), Some(2));
    ///
    /// // indices 1, 3
    /// assert_eq!(Slice::new(1, Some(5), 2).nth(1), Some(3));
    ///
    /// // indices 1, -1, -3
    /// assert_eq!(Slice::new(1, Some(-4), -2).nth(2), Some(-3));
    ///
    /// // indices 1, 3, 5, ...
    /// assert_eq!(Slice::new(1, None, 2).nth(2), Some(5));
    ///
    /// // indices 3, 3, 3, ...
    /// assert_eq!(Slice::new(3, Some(5), 0).nth(1000), Some(3));
    ///
    /// // indices 1, 3
    /// assert_eq!(Slice::new(1, Some(5), 2).nth(2), None);
    /// ```
    pub fn nth(&self, n: usize) -> Option<isize> {
        let nth = self.start + self.step * n as isize;
        if let Some(end) = self.end {
            if self.step == 0 || (self.step > 0 && nth < end) || (self.step < 0 && nth > end) {
                Some(nth)
            } else {
                None
            }
        } else {
            Some(nth)
        }
    }
}

impl From<Range<Ixs>> for Slice {
    #[inline]
    fn from(r: Range<Ixs>) -> Slice {
        Slice {
            start: r.start,
            end: Some(r.end),
            step: 1,
        }
    }
}

impl From<RangeFrom<Ixs>> for Slice {
    #[inline]
    fn from(r: RangeFrom<Ixs>) -> Slice {
        Slice {
            start: r.start,
            end: None,
            step: 1,
        }
    }
}

impl From<RangeTo<Ixs>> for Slice {
    #[inline]
    fn from(r: RangeTo<Ixs>) -> Slice {
        Slice {
            start: 0,
            end: Some(r.end),
            step: 1,
        }
    }
}

impl From<RangeFull> for Slice {
    #[inline]
    fn from(_: RangeFull) -> Slice {
        Slice {
            start: 0,
            end: None,
            step: 1,
        }
    }
}

/// A slice (range with step) or an index.
///
/// See also the [`s![]`](macro.s!.html) macro for a convenient way to create a
/// `&SliceInfo<[SliceOrIndex; n], D>`.
///
/// ## Examples
///
/// `SliceOrIndex::Index(a)` is the index `a`. It can also be created with
/// `SliceOrIndex::from(a)`. The Python equivalent is `[a]`. The macro
/// equivalent is `s![a]`.
///
/// `SliceOrIndex::Slice { start: 0, end: None, step: 1 }` is the full range of
/// an axis. It can also be created with `SliceOrIndex::from(..)`. The Python
/// equivalent is `[:]`. The macro equivalent is `s![..]`.
///
/// `SliceOrIndex::Slice { start: a, end: Some(b), step: 2 }` is every second
/// element from `a` until `b`. It can also be created with
/// `SliceOrIndex::from(a..b).step_by(2)`. The Python equivalent is `[a:b:2]`.
/// The macro equivalent is `s![a..b;2]`.
///
/// `SliceOrIndex::Slice { start: a, end: None, step: -1 }` is every element,
/// from `a` until the end, in reverse order. It can also be created with
/// `SliceOrIndex::from(a..).step_by(-1)`. The Python equivalent is `[a::-1]`.
/// The macro equivalent is `s![a..;-1]`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum SliceOrIndex {
    /// A range with step size. Negative `begin` or `end` indexes are counted
    /// from the back of the axis. If `end` is `None`, the slice extends to the
    /// end of the axis.
    Slice {
        start: Ixs,
        end: Option<Ixs>,
        step: Ixs,
    },
    /// A single index.
    Index(Ixs),
}

copy_and_clone!{SliceOrIndex}

impl SliceOrIndex {
    /// Returns `true` if `self` is a `Slice` value.
    pub fn is_slice(&self) -> bool {
        match self {
            &SliceOrIndex::Slice { .. } => true,
            _ => false,
        }
    }

    /// Returns `true` if `self` is an `Index` value.
    pub fn is_index(&self) -> bool {
        match self {
            &SliceOrIndex::Index(_) => true,
            _ => false,
        }
    }

    /// Returns a new `SliceOrIndex` with the given step size.
    #[inline]
    pub fn step_by(self, step: Ixs) -> Self {
        match self {
            SliceOrIndex::Slice { start, end, .. } => SliceOrIndex::Slice { start, end, step },
            SliceOrIndex::Index(s) => SliceOrIndex::Index(s),
        }
    }
}

impl fmt::Display for SliceOrIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SliceOrIndex::Index(index) => write!(f, "{}", index)?,
            SliceOrIndex::Slice { start, end, step } => {
                if start != 0 {
                    write!(f, "{}", start)?;
                }
                write!(f, "..")?;
                if let Some(i) = end {
                    write!(f, "{}", i)?;
                }
                if step != 1 {
                    write!(f, ";{}", step)?;
                }
            }
        }
        Ok(())
    }
}

impl From<Slice> for SliceOrIndex {
    #[inline]
    fn from(s: Slice) -> SliceOrIndex {
        SliceOrIndex::Slice {
            start: s.start,
            end: s.end,
            step: s.step,
        }
    }
}

impl From<Range<Ixs>> for SliceOrIndex {
    #[inline]
    fn from(r: Range<Ixs>) -> SliceOrIndex {
        SliceOrIndex::Slice {
            start: r.start,
            end: Some(r.end),
            step: 1,
        }
    }
}

impl From<Ixs> for SliceOrIndex {
    #[inline]
    fn from(r: Ixs) -> SliceOrIndex {
        SliceOrIndex::Index(r)
    }
}

impl From<RangeFrom<Ixs>> for SliceOrIndex {
    #[inline]
    fn from(r: RangeFrom<Ixs>) -> SliceOrIndex {
        SliceOrIndex::Slice {
            start: r.start,
            end: None,
            step: 1,
        }
    }
}

impl From<RangeTo<Ixs>> for SliceOrIndex {
    #[inline]
    fn from(r: RangeTo<Ixs>) -> SliceOrIndex {
        SliceOrIndex::Slice {
            start: 0,
            end: Some(r.end),
            step: 1,
        }
    }
}

impl From<RangeFull> for SliceOrIndex {
    #[inline]
    fn from(_: RangeFull) -> SliceOrIndex {
        SliceOrIndex::Slice {
            start: 0,
            end: None,
            step: 1,
        }
    }
}

/// Represents all of the necessary information to perform a slice.
///
/// The type `T` is typically `[SliceOrIndex; n]`, `[SliceOrIndex]`, or
/// `Vec<SliceOrIndex>`. The type `D` is the output dimension after calling
/// [`.slice()`].
///
/// [`.slice()`]: struct.ArrayBase.html#method.slice
#[derive(Debug)]
#[repr(C)]
pub struct SliceInfo<T: ?Sized, D: Dimension> {
    out_dim: PhantomData<D>,
    indices: T,
}

impl<T: ?Sized, D> Deref for SliceInfo<T, D>
where
    D: Dimension,
{
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.indices
    }
}

impl<T, D> SliceInfo<T, D>
where
    D: Dimension,
{
    /// Returns a new `SliceInfo` instance.
    ///
    /// If you call this method, you are guaranteeing that `out_dim` and
    /// `out_ndim` are consistent with `indices`.
    #[doc(hidden)]
    pub unsafe fn new_unchecked(indices: T, out_dim: PhantomData<D>) -> SliceInfo<T, D> {
        SliceInfo {
            out_dim: out_dim,
            indices: indices,
        }
    }
}

impl<T, D> SliceInfo<T, D>
where
    T: AsRef<[SliceOrIndex]>,
    D: Dimension,
{
    /// Returns a new `SliceInfo` instance.
    ///
    /// **Panics** if `D` is not consistent with `indices`.
    pub fn new(indices: T) -> SliceInfo<T, D> {
        let out_ndim = indices.as_ref().iter().filter(|s| s.is_slice()).count();
        if let Some(ndim) = D::NDIM {
            assert_eq!(out_ndim, ndim);
        }
        SliceInfo {
            out_dim: PhantomData,
            indices: indices,
        }
    }
}

impl<T: ?Sized, D> SliceInfo<T, D>
where
    T: AsRef<[SliceOrIndex]>,
    D: Dimension,
{
    /// Returns the number of dimensions after calling
    /// [`.slice()`](struct.ArrayBase.html#method.slice) (including taking
    /// subviews).
    ///
    /// If `D` is a fixed-size dimension type, then this is equivalent to
    /// `D::NDIM.unwrap()`. Otherwise, the value is calculated by iterating
    /// over the ranges/indices.
    pub fn out_ndim(&self) -> usize {
        D::NDIM.unwrap_or_else(|| {
            self.indices
                .as_ref()
                .iter()
                .filter(|s| s.is_slice())
                .count()
        })
    }
}

impl<T, D> AsRef<[SliceOrIndex]> for SliceInfo<T, D>
where
    T: AsRef<[SliceOrIndex]>,
    D: Dimension,
{
    fn as_ref(&self) -> &[SliceOrIndex] {
        self.indices.as_ref()
    }
}

impl<T, D> AsRef<SliceInfo<[SliceOrIndex], D>> for SliceInfo<T, D>
where
    T: AsRef<[SliceOrIndex]>,
    D: Dimension,
{
    fn as_ref(&self) -> &SliceInfo<[SliceOrIndex], D> {
        unsafe {
            // This is okay because the only non-zero-sized member of
            // `SliceInfo` is `indices`, so `&SliceInfo<[SliceOrIndex], D>`
            // should have the same bitwise representation as
            // `&[SliceOrIndex]`.
            &*(self.indices.as_ref() as *const [SliceOrIndex]
                as *const SliceInfo<[SliceOrIndex], D>)
        }
    }
}

impl<T, D> Copy for SliceInfo<T, D>
where
    T: Copy,
    D: Dimension,
{
}

impl<T, D> Clone for SliceInfo<T, D>
where
    T: Clone,
    D: Dimension,
{
    fn clone(&self) -> Self {
        SliceInfo {
            out_dim: PhantomData,
            indices: self.indices.clone(),
        }
    }
}


#[doc(hidden)]
pub trait SliceNextDim<D1, D2> {
    fn next_dim(&self, PhantomData<D1>) -> PhantomData<D2>;
}

impl<D1: Dimension> SliceNextDim<D1, D1::Larger> for Slice {
    fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
        PhantomData
    }
}

impl<D1: Dimension> SliceNextDim<D1, D1::Larger> for Range<Ixs> {
    fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
        PhantomData
    }
}

impl<D1: Dimension> SliceNextDim<D1, D1::Larger> for RangeFrom<Ixs> {
    fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
        PhantomData
    }
}

impl<D1: Dimension> SliceNextDim<D1, D1::Larger> for RangeTo<Ixs> {
    fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
        PhantomData
    }
}

impl<D1: Dimension> SliceNextDim<D1, D1::Larger> for RangeFull {
    fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
        PhantomData
    }
}

impl<D1: Dimension> SliceNextDim<D1, D1> for Ixs {
    fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1> {
        PhantomData
    }
}

/// Slice argument constructor.
///
/// `s![]` takes a list of ranges/slices/indices, separated by comma, with
/// optional step sizes that are separated from the range by a semicolon. It is
/// converted into a [`&SliceInfo`] instance.
///
/// [`&SliceInfo`]: struct.SliceInfo.html
///
/// Each range/slice/index uses signed indices, where a negative value is
/// counted from the end of the axis. Step sizes are also signed and may be
/// negative, but must not be zero.
///
/// The syntax is `s![` *[ axis-slice-or-index [, axis-slice-or-index [ , ... ]
/// ] ]* `]`, where *axis-slice-or-index* is any of the following:
///
/// * *index*: an index to use for taking a subview with respect to that axis
/// * *range*: a range with step size 1 to use for slicing that axis
/// * *range* `;` *step*: a range with step size *step* to use for slicing that axis
/// * *slice*: a [`Slice`] instance to use for slicing that axis
/// * *slice* `;` *step*: a range constructed from the start and end of a [`Slice`]
///   instance, with new step size *step*, to use for slicing that axis
///
/// [`Slice`]: struct.Slice.html
///
/// The number of *axis-slice-or-index* must match the number of axes in the
/// array. *index*, *range*, *slice*, and *step* can be expressions. *index*
/// and *step* must be of type [`Ixs`]. *range* can be of type `Range<Ixs>`,
/// `RangeTo<Ixs>`, `RangeFrom<Ixs>`, or `RangeFull`.
///
/// [`Ixs`]: type.Ixs.html
///
/// For example `s![0..4;2, 6, 1..5]` is a slice of the first axis for 0..4
/// with step size 2, a subview of the second axis at index 6, and a slice of
/// the third axis for 1..5 with default step size 1. The input array must have
/// 3 dimensions. The resulting slice would have shape `[2, 4]` for
/// [`.slice()`], [`.slice_mut()`], and [`.slice_move()`], and shape
/// `[2, 1, 4]` for [`.slice_inplace()`].
///
/// [`.slice()`]: struct.ArrayBase.html#method.slice
/// [`.slice_mut()`]: struct.ArrayBase.html#method.slice_mut
/// [`.slice_move()`]: struct.ArrayBase.html#method.slice_move
/// [`.slice_inplace()`]: struct.ArrayBase.html#method.slice_inplace
///
/// See also [*Slicing*](struct.ArrayBase.html#slicing).
///
/// # Example
///
/// ```
/// #[macro_use]
/// extern crate ndarray;
///
/// use ndarray::{Array2, ArrayView2};
///
/// fn laplacian(v: &ArrayView2<f32>) -> Array2<f32> {
///     -4. * &v.slice(s![1..-1, 1..-1])
///     + v.slice(s![ ..-2, 1..-1])
///     + v.slice(s![1..-1,  ..-2])
///     + v.slice(s![1..-1, 2..  ])
///     + v.slice(s![2..  , 1..-1])
/// }
/// # fn main() { }
/// ```
#[macro_export]
macro_rules! s(
    // convert a..b;c into @convert(a..b, c), final item
    (@parse $dim:expr, [$($stack:tt)*] $r:expr;$s:expr) => {
        unsafe {
            &$crate::SliceInfo::new_unchecked(
                [$($stack)* s!(@convert $r, $s)],
                $crate::SliceNextDim::next_dim(&$r, $dim),
            )
        }
    };
    // convert a..b into @convert(a..b), final item
    (@parse $dim:expr, [$($stack:tt)*] $r:expr) => {
        unsafe {
            &$crate::SliceInfo::new_unchecked(
                [$($stack)* s!(@convert $r)],
                $crate::SliceNextDim::next_dim(&$r, $dim),
            )
        }
    };
    // convert a..b;c into @convert(a..b, c), final item, trailing comma
    (@parse $dim:expr, [$($stack:tt)*] $r:expr;$s:expr ,) => {
        unsafe {
            &$crate::SliceInfo::new_unchecked(
                [$($stack)* s!(@convert $r, $s)],
                $crate::SliceNextDim::next_dim(&$r, $dim),
            )
        }
    };
    // convert a..b into @convert(a..b), final item, trailing comma
    (@parse $dim:expr, [$($stack:tt)*] $r:expr ,) => {
        unsafe {
            &$crate::SliceInfo::new_unchecked(
                [$($stack)* s!(@convert $r)],
                $crate::SliceNextDim::next_dim(&$r, $dim),
            )
        }
    };
    // convert a..b;c into @convert(a..b, c)
    (@parse $dim:expr, [$($stack:tt)*] $r:expr;$s:expr, $($t:tt)*) => {
        s![@parse
            $crate::SliceNextDim::next_dim(&$r, $dim),
            [$($stack)* s!(@convert $r, $s),]
            $($t)*
        ]
    };
    // convert a..b into @convert(a..b)
    (@parse $dim:expr, [$($stack:tt)*] $r:expr, $($t:tt)*) => {
        s![@parse
            $crate::SliceNextDim::next_dim(&$r, $dim),
            [$($stack)* s!(@convert $r),]
            $($t)*
        ]
    };
    // convert range/index into SliceOrIndex
    (@convert $r:expr) => {
        <$crate::SliceOrIndex as ::std::convert::From<_>>::from($r)
    };
    // convert range/index and step into SliceOrIndex
    (@convert $r:expr, $s:expr) => {
        <$crate::SliceOrIndex as ::std::convert::From<_>>::from($r).step_by($s)
    };
    ($($t:tt)*) => {
        s![@parse ::std::marker::PhantomData::<$crate::Ix0>, [] $($t)*]
    };
);
