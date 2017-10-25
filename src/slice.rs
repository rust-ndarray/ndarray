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

/// A slice (range with step) or an index.
///
///
/// For the `Slice` variant, the fields are `begin`, `end`, and `step`, where
/// negative `begin` or `end` indexes are counted from the back of the axis. If
/// `end` is `None`, the slice extends to the end of the axis.
///
/// For the `Index` variant, the field is the index.
///
/// See also the [`s![] macro`](macro.s!.html) for a convenient way to create a
/// `&SliceInfo<[SliceOrIndex; n], D>`.
///
/// ## Examples
///
/// `SliceOrIndex::Slice(0, None, 1)` is the full range of an axis. It can also
/// be created with `SliceOrIndex::from(..)`. The Python equivalent is `[:]`.
/// The macro equivalent is `s![..]`.
///
/// `SliceOrIndex::Slice(a, Some(b), 2)` is every second element from `a` until
/// `b`. It can also be created with `SliceOrIndex::from(a..b).step(2)`. The
/// Python equivalent is `[a:b:2]`. The macro equivalent is `s![a..b;2]`.
///
/// `SliceOrIndex::Slice(a, None, -1)` is every element, from `a` until the
/// end, in reverse order. It can also be created with
/// `SliceOrIndex::from(a..).step(-1)`. The Python equivalent is `[a::-1]`. The
/// macro equivalent is `s![a..;-1]`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum SliceOrIndex {
    Slice(Ixs, Option<Ixs>, Ixs),
    Index(Ixs),
}

copy_and_clone!{SliceOrIndex}

impl SliceOrIndex {
    pub fn is_slice(&self) -> bool {
        match self {
            &SliceOrIndex::Slice(..) => true,
            _ => false,
        }
    }

    pub fn is_index(&self) -> bool {
        match self {
            &SliceOrIndex::Index(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn step(self, step: Ixs) -> Self {
        match self {
            SliceOrIndex::Slice(start, end, _) => SliceOrIndex::Slice(start, end, step),
            SliceOrIndex::Index(s) => SliceOrIndex::Index(s),
        }
    }
}

impl fmt::Display for SliceOrIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SliceOrIndex::Index(index) => write!(f, "{}", index)?,
            SliceOrIndex::Slice(start, end, step) => {
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

impl From<Range<Ixs>> for SliceOrIndex {
    #[inline]
    fn from(r: Range<Ixs>) -> SliceOrIndex {
        SliceOrIndex::Slice(r.start, Some(r.end), 1)
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
        SliceOrIndex::Slice(r.start, None, 1)
    }
}

impl From<RangeTo<Ixs>> for SliceOrIndex {
    #[inline]
    fn from(r: RangeTo<Ixs>) -> SliceOrIndex {
        SliceOrIndex::Slice(0, Some(r.end), 1)
    }
}

impl From<RangeFull> for SliceOrIndex {
    #[inline]
    fn from(_: RangeFull) -> SliceOrIndex {
        SliceOrIndex::Slice(0, None, 1)
    }
}

/// Represents all of the necessary information to perform a slice.
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
    /// Returns the number of dimensions after slicing and taking subviews.
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
/// `s![]` takes a list of ranges, separated by comma, with optional strides
/// that are separated from the range by a semicolon. It is converted into a
/// `SliceInfo` instance.
///
/// Each range uses signed indices, where a negative value is counted from
/// the end of the axis. Strides are also signed and may be negative, but
/// must not be zero.
///
/// The syntax is `s![` *[ axis-slice [, axis-slice [ , ... ] ] ]* `]`.
/// Where *axis-slice* is either *i* `..` *j* or *i* `..` *j* `;` *step*,
/// and *i* is the start index, *j* end index and *step* the element step
/// size (which defaults to 1). The number of *axis-slice* must match the
/// number of axes in the array.
///
/// For example `s![0..4;2, 1..5]` is a slice of rows 0..4 with step size 2,
/// and columns 1..5 with default step size 1. The slice would have
/// shape `[2, 4]`.
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
    // convert a..b;c into @step(a..b, c), final item
    (@parse $dim:expr, [$($stack:tt)*] $r:expr;$s:expr) => {
        unsafe {
            &$crate::SliceInfo::new_unchecked(
                [$($stack)* s!(@step $r, $s)],
                $crate::SliceNextDim::next_dim(&$r, $dim),
            )
        }
    };
    // convert a..b into @step(a..b, 1), final item
    (@parse $dim:expr, [$($stack:tt)*] $r:expr) => {
        unsafe {
            &$crate::SliceInfo::new_unchecked(
                [$($stack)* s!(@step $r, 1)],
                $crate::SliceNextDim::next_dim(&$r, $dim),
            )
        }
    };
    // convert a..b;c into @step(a..b, c), final item, trailing comma
    (@parse $dim:expr, [$($stack:tt)*] $r:expr;$s:expr ,) => {
        unsafe {
            &$crate::SliceInfo::new_unchecked(
                [$($stack)* s!(@step $r, $s)],
                $crate::SliceNextDim::next_dim(&$r, $dim),
            )
        }
    };
    // convert a..b into @step(a..b, 1), final item, trailing comma
    (@parse $dim:expr, [$($stack:tt)*] $r:expr ,) => {
        unsafe {
            &$crate::SliceInfo::new_unchecked(
                [$($stack)* s!(@step $r, 1)],
                $crate::SliceNextDim::next_dim(&$r, $dim),
            )
        }
    };
    // convert a..b;c into @step(a..b, c)
    (@parse $dim:expr, [$($stack:tt)*] $r:expr;$s:expr, $($t:tt)*) => {
        s![@parse $crate::SliceNextDim::next_dim(&$r, $dim), [$($stack)* s!(@step $r, $s),] $($t)*]
    };
    // convert a..b into @step(a..b, 1)
    (@parse $dim:expr, [$($stack:tt)*] $r:expr, $($t:tt)*) => {
        s![@parse $crate::SliceNextDim::next_dim(&$r, $dim), [$($stack)* s!(@step $r, 1),] $($t)*]
    };
    // convert range, step into SliceOrIndex
    (@step $r:expr, $s:expr) => {
        <$crate::SliceOrIndex as ::std::convert::From<_>>::from($r).step($s)
    };
    ($($t:tt)*) => {
        s![@parse ::std::marker::PhantomData::<$crate::Ix0>, [] $($t)*]
    };
);
