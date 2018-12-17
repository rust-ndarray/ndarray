// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use crate::dimension::slices_intersect;
use crate::error::{ErrorKind, ShapeError};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use crate::{ArrayViewMut, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

/// A slice (range with step size).
///
/// `end` is an exclusive index. Negative `begin` or `end` indexes are counted
/// from the back of the axis. If `end` is `None`, the slice extends to the end
/// of the axis.
///
/// See also the [`s![]`](macro.s.html) macro.
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
    /// Create a new `Slice` with the given extents.
    ///
    /// See also the `From` impls, converting from ranges; for example
    /// `Slice::from(i..)` or `Slice::from(j..k)`.
    ///
    /// `step` must be nonzero.
    /// (This method checks with a debug assertion that `step` is not zero.)
    pub fn new(start: isize, end: Option<isize>, step: isize) -> Slice {
        debug_assert_ne!(step, 0, "Slice::new: step must be nonzero");
        Slice { start, end, step }
    }

    /// Create a new `Slice` with the given step size (multiplied with the
    /// previous step size).
    ///
    /// `step` must be nonzero.
    /// (This method checks with a debug assertion that `step` is not zero.)
    #[inline]
    pub fn step_by(self, step: isize) -> Self {
        debug_assert_ne!(step, 0, "Slice::step_by: step must be nonzero");
        Slice {
            step: self.step * step,
            ..self
        }
    }
}

/// Token to represent a new axis in a slice description.
///
/// See also the [`s![]`](macro.s!.html) macro.
pub struct NewAxis;

/// A slice (range with step), an index, or a new axis token.
///
/// See also the [`s![]`](macro.s!.html) macro for a convenient way to create a
/// `&SliceInfo<[AxisSliceInfo; n], Din, Dout>`.
///
/// ## Examples
///
/// `AxisSliceInfo::Index(a)` is the index `a`. It can also be created with
/// `AxisSliceInfo::from(a)`. The Python equivalent is `[a]`. The macro
/// equivalent is `s![a]`.
///
/// `AxisSliceInfo::Slice { start: 0, end: None, step: 1 }` is the full range
/// of an axis. It can also be created with `AxisSliceInfo::from(..)`. The
/// Python equivalent is `[:]`. The macro equivalent is `s![..]`.
///
/// `AxisSliceInfo::Slice { start: a, end: Some(b), step: 2 }` is every second
/// element from `a` until `b`. It can also be created with
/// `AxisSliceInfo::from(a..b).step_by(2)`. The Python equivalent is `[a:b:2]`.
/// The macro equivalent is `s![a..b;2]`.
///
/// `AxisSliceInfo::Slice { start: a, end: None, step: -1 }` is every element,
/// from `a` until the end, in reverse order. It can also be created with
/// `AxisSliceInfo::from(a..).step_by(-1)`. The Python equivalent is `[a::-1]`.
/// The macro equivalent is `s![a..;-1]`.
///
/// `AxisSliceInfo::NewAxis` is a new axis of length 1. It can also be created
/// with `AxisSliceInfo::from(NewAxis)`. The Python equivalent is
/// `[np.newaxis]`. The macro equivalent is `s![NewAxis]`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum AxisSliceInfo {
    /// A range with step size. `end` is an exclusive index. Negative `begin`
    /// or `end` indexes are counted from the back of the axis. If `end` is
    /// `None`, the slice extends to the end of the axis.
    Slice {
        start: isize,
        end: Option<isize>,
        step: isize,
    },
    /// A single index.
    Index(isize),
    /// A new axis of length 1.
    NewAxis,
}

copy_and_clone! {AxisSliceInfo}

impl AxisSliceInfo {
    /// Returns `true` if `self` is a `Slice` value.
    pub fn is_slice(&self) -> bool {
        matches!(self, AxisSliceInfo::Slice { .. })
    }

    /// Returns `true` if `self` is an `Index` value.
    pub fn is_index(&self) -> bool {
        matches!(self, AxisSliceInfo::Index(_))
    }

    /// Returns `true` if `self` is a `NewAxis` value.
    pub fn is_new_axis(&self) -> bool {
        matches!(self, AxisSliceInfo::NewAxis)
    }

    /// Returns a new `AxisSliceInfo` with the given step size (multiplied with
    /// the previous step size).
    ///
    /// `step` must be nonzero.
    /// (This method checks with a debug assertion that `step` is not zero.)
    #[inline]
    pub fn step_by(self, step: isize) -> Self {
        debug_assert_ne!(step, 0, "AxisSliceInfo::step_by: step must be nonzero");
        match self {
            AxisSliceInfo::Slice {
                start,
                end,
                step: orig_step,
            } => AxisSliceInfo::Slice {
                start,
                end,
                step: orig_step * step,
            },
            AxisSliceInfo::Index(s) => AxisSliceInfo::Index(s),
            AxisSliceInfo::NewAxis => AxisSliceInfo::NewAxis,
        }
    }
}

impl fmt::Display for AxisSliceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            AxisSliceInfo::Index(index) => write!(f, "{}", index)?,
            AxisSliceInfo::Slice { start, end, step } => {
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
            AxisSliceInfo::NewAxis => write!(f, "NewAxis")?,
        }
        Ok(())
    }
}

macro_rules! impl_slice_variant_from_range {
    ($self:ty, $constructor:path, $index:ty) => {
        impl From<Range<$index>> for $self {
            #[inline]
            fn from(r: Range<$index>) -> $self {
                $constructor {
                    start: r.start as isize,
                    end: Some(r.end as isize),
                    step: 1,
                }
            }
        }

        impl From<RangeInclusive<$index>> for $self {
            #[inline]
            fn from(r: RangeInclusive<$index>) -> $self {
                let end = *r.end() as isize;
                $constructor {
                    start: *r.start() as isize,
                    end: if end == -1 { None } else { Some(end + 1) },
                    step: 1,
                }
            }
        }

        impl From<RangeFrom<$index>> for $self {
            #[inline]
            fn from(r: RangeFrom<$index>) -> $self {
                $constructor {
                    start: r.start as isize,
                    end: None,
                    step: 1,
                }
            }
        }

        impl From<RangeTo<$index>> for $self {
            #[inline]
            fn from(r: RangeTo<$index>) -> $self {
                $constructor {
                    start: 0,
                    end: Some(r.end as isize),
                    step: 1,
                }
            }
        }

        impl From<RangeToInclusive<$index>> for $self {
            #[inline]
            fn from(r: RangeToInclusive<$index>) -> $self {
                let end = r.end as isize;
                $constructor {
                    start: 0,
                    end: if end == -1 { None } else { Some(end + 1) },
                    step: 1,
                }
            }
        }
    };
}
impl_slice_variant_from_range!(Slice, Slice, isize);
impl_slice_variant_from_range!(Slice, Slice, usize);
impl_slice_variant_from_range!(Slice, Slice, i32);
impl_slice_variant_from_range!(AxisSliceInfo, AxisSliceInfo::Slice, isize);
impl_slice_variant_from_range!(AxisSliceInfo, AxisSliceInfo::Slice, usize);
impl_slice_variant_from_range!(AxisSliceInfo, AxisSliceInfo::Slice, i32);

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

impl From<RangeFull> for AxisSliceInfo {
    #[inline]
    fn from(_: RangeFull) -> AxisSliceInfo {
        AxisSliceInfo::Slice {
            start: 0,
            end: None,
            step: 1,
        }
    }
}

impl From<Slice> for AxisSliceInfo {
    #[inline]
    fn from(s: Slice) -> AxisSliceInfo {
        AxisSliceInfo::Slice {
            start: s.start,
            end: s.end,
            step: s.step,
        }
    }
}

macro_rules! impl_axissliceinfo_from_index {
    ($index:ty) => {
        impl From<$index> for AxisSliceInfo {
            #[inline]
            fn from(r: $index) -> AxisSliceInfo {
                AxisSliceInfo::Index(r as isize)
            }
        }
    };
}
impl_axissliceinfo_from_index!(isize);
impl_axissliceinfo_from_index!(usize);
impl_axissliceinfo_from_index!(i32);

impl From<NewAxis> for AxisSliceInfo {
    #[inline]
    fn from(_: NewAxis) -> AxisSliceInfo {
        AxisSliceInfo::NewAxis
    }
}

/// A type that can slice an array of dimension `D`.
///
/// This trait is unsafe to implement because the implementation must ensure
/// that `D`, `Self::OutDim`, `self.in_dim()`, and `self.out_ndim()` are
/// consistent with the `&[AxisSliceInfo]` returned by `self.as_ref()`.
pub unsafe trait CanSlice<D: Dimension>: AsRef<[AxisSliceInfo]> {
    type OutDim: Dimension;

    fn in_ndim(&self) -> usize;

    fn out_ndim(&self) -> usize;
}

macro_rules! impl_canslice_samedim {
    ($in_dim:ty) => {
        unsafe impl<T, Dout> CanSlice<$in_dim> for SliceInfo<T, $in_dim, Dout>
        where
            T: AsRef<[AxisSliceInfo]>,
            Dout: Dimension,
        {
            type OutDim = Dout;

            fn in_ndim(&self) -> usize {
                self.in_ndim()
            }

            fn out_ndim(&self) -> usize {
                self.out_ndim()
            }
        }
    };
}
impl_canslice_samedim!(Ix0);
impl_canslice_samedim!(Ix1);
impl_canslice_samedim!(Ix2);
impl_canslice_samedim!(Ix3);
impl_canslice_samedim!(Ix4);
impl_canslice_samedim!(Ix5);
impl_canslice_samedim!(Ix6);

unsafe impl<T, Din, Dout> CanSlice<IxDyn> for SliceInfo<T, Din, Dout>
where
    T: AsRef<[AxisSliceInfo]>,
    Din: Dimension,
    Dout: Dimension,
{
    type OutDim = Dout;

    fn in_ndim(&self) -> usize {
        self.in_ndim()
    }

    fn out_ndim(&self) -> usize {
        self.out_ndim()
    }
}

/// Represents all of the necessary information to perform a slice.
///
/// The type `T` is typically `[AxisSliceInfo; n]`, `[AxisSliceInfo]`, or
/// `Vec<AxisSliceInfo>`. The type `Din` is the dimension of the array to be
/// sliced, and `Dout` is the output dimension after calling [`.slice()`]. Note
/// that if `Din` is a fixed dimension type (`Ix0`, `Ix1`, `Ix2`, etc.), the
/// `SliceInfo` instance can still be used to slice an array with dimension
/// `IxDyn` as long as the number of axes matches.
///
/// [`.slice()`]: struct.ArrayBase.html#method.slice
#[derive(Debug)]
#[repr(C)]
pub struct SliceInfo<T: ?Sized, Din: Dimension, Dout: Dimension> {
    in_dim: PhantomData<Din>,
    out_dim: PhantomData<Dout>,
    indices: T,
}

impl<T: ?Sized, Din, Dout> Deref for SliceInfo<T, Din, Dout>
where
    Din: Dimension,
    Dout: Dimension,
{
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.indices
    }
}

impl<T, Din, Dout> SliceInfo<T, Din, Dout>
where
    Din: Dimension,
    Dout: Dimension,
{
    /// Returns a new `SliceInfo` instance.
    ///
    /// If you call this method, you are guaranteeing that `in_dim` and
    /// `out_dim` are consistent with `indices`.
    #[doc(hidden)]
    pub unsafe fn new_unchecked(
        indices: T,
        in_dim: PhantomData<Din>,
        out_dim: PhantomData<Dout>,
    ) -> SliceInfo<T, Din, Dout> {
        SliceInfo {
            in_dim: in_dim,
            out_dim: out_dim,
            indices: indices,
        }
    }
}

impl<T, Din, Dout> SliceInfo<T, Din, Dout>
where
    T: AsRef<[AxisSliceInfo]>,
    Din: Dimension,
    Dout: Dimension,
{
    /// Returns a new `SliceInfo` instance.
    ///
    /// Errors if `Din` or `Dout` is not consistent with `indices`.
    pub fn new(indices: T) -> Result<SliceInfo<T, Din, Dout>, ShapeError> {
        if let Some(ndim) = Din::NDIM {
            if ndim != indices.as_ref().iter().filter(|s| !s.is_new_axis()).count() {
                return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
            }
        }
        if let Some(ndim) = Dout::NDIM {
            if ndim != indices.as_ref().iter().filter(|s| !s.is_index()).count() {
                return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
            }
        }
        Ok(SliceInfo {
            in_dim: PhantomData,
            out_dim: PhantomData,
            indices,
        })
    }
}

impl<T: ?Sized, Din, Dout> SliceInfo<T, Din, Dout>
where
    T: AsRef<[AxisSliceInfo]>,
    Din: Dimension,
    Dout: Dimension,
{
    /// Returns the number of dimensions of the input array for
    /// [`.slice()`](struct.ArrayBase.html#method.slice).
    ///
    /// If `Din` is a fixed-size dimension type, then this is equivalent to
    /// `Din::NDIM.unwrap()`. Otherwise, the value is calculated by iterating
    /// over the `AxisSliceInfo` elements.
    pub fn in_ndim(&self) -> usize {
        Din::NDIM.unwrap_or_else(|| {
            self.indices
                .as_ref()
                .iter()
                .filter(|s| !s.is_new_axis())
                .count()
        })
    }

    /// Returns the number of dimensions after calling
    /// [`.slice()`](struct.ArrayBase.html#method.slice) (including taking
    /// subviews).
    ///
    /// If `Dout` is a fixed-size dimension type, then this is equivalent to
    /// `Dout::NDIM.unwrap()`. Otherwise, the value is calculated by iterating
    /// over the `AxisSliceInfo` elements.
    pub fn out_ndim(&self) -> usize {
        Dout::NDIM.unwrap_or_else(|| {
            self.indices
                .as_ref()
                .iter()
                .filter(|s| !s.is_index())
                .count()
        })
    }
}

impl<T, Din, Dout> AsRef<[AxisSliceInfo]> for SliceInfo<T, Din, Dout>
where
    T: AsRef<[AxisSliceInfo]>,
    Din: Dimension,
    Dout: Dimension,
{
    fn as_ref(&self) -> &[AxisSliceInfo] {
        self.indices.as_ref()
    }
}

impl<T, Din, Dout> AsRef<SliceInfo<[AxisSliceInfo], Din, Dout>> for SliceInfo<T, Din, Dout>
where
    T: AsRef<[AxisSliceInfo]>,
    Din: Dimension,
    Dout: Dimension,
{
    fn as_ref(&self) -> &SliceInfo<[AxisSliceInfo], Din, Dout> {
        unsafe {
            // This is okay because the only non-zero-sized member of
            // `SliceInfo` is `indices`, so `&SliceInfo<[AxisSliceInfo], Din, Dout>`
            // should have the same bitwise representation as
            // `&[AxisSliceInfo]`.
            &*(self.indices.as_ref() as *const [AxisSliceInfo]
                as *const SliceInfo<[AxisSliceInfo], Din, Dout>)
        }
    }
}

impl<T, Din, Dout> Copy for SliceInfo<T, Din, Dout>
where
    T: Copy,
    Din: Dimension,
    Dout: Dimension,
{
}

impl<T, Din, Dout> Clone for SliceInfo<T, Din, Dout>
where
    T: Clone,
    Din: Dimension,
    Dout: Dimension,
{
    fn clone(&self) -> Self {
        SliceInfo {
            in_dim: PhantomData,
            out_dim: PhantomData,
            indices: self.indices.clone(),
        }
    }
}

#[doc(hidden)]
pub trait SliceNextInDim<D1, D2> {
    fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D2>;
}

impl<D1: Dimension> SliceNextInDim<D1, D1> for NewAxis {
    fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1> {
        PhantomData
    }
}

macro_rules! impl_slicenextindim_larger {
    (($($generics:tt)*), $self:ty) => {
        impl<D1: Dimension, $($generics),*> SliceNextInDim<D1, D1::Larger> for $self {
            fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
                PhantomData
            }
        }
    }
}
impl_slicenextindim_larger!((), isize);
impl_slicenextindim_larger!((), usize);
impl_slicenextindim_larger!((), i32);
impl_slicenextindim_larger!((T), Range<T>);
impl_slicenextindim_larger!((T), RangeInclusive<T>);
impl_slicenextindim_larger!((T), RangeFrom<T>);
impl_slicenextindim_larger!((T), RangeTo<T>);
impl_slicenextindim_larger!((T), RangeToInclusive<T>);
impl_slicenextindim_larger!((), RangeFull);
impl_slicenextindim_larger!((), Slice);

#[doc(hidden)]
pub trait SliceNextOutDim<D1, D2> {
    fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D2>;
}

macro_rules! impl_slicenextoutdim_equal {
    ($self:ty) => {
        impl<D1: Dimension> SliceNextOutDim<D1, D1> for $self {
            fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1> {
                PhantomData
            }
        }
    };
}
impl_slicenextoutdim_equal!(isize);
impl_slicenextoutdim_equal!(usize);
impl_slicenextoutdim_equal!(i32);

macro_rules! impl_slicenextoutdim_larger {
    (($($generics:tt)*), $self:ty) => {
        impl<D1: Dimension, $($generics)*> SliceNextOutDim<D1, D1::Larger> for $self {
            fn next_dim(&self, _: PhantomData<D1>) -> PhantomData<D1::Larger> {
                PhantomData
            }
        }
    }
}
impl_slicenextoutdim_larger!((T), Range<T>);
impl_slicenextoutdim_larger!((T), RangeInclusive<T>);
impl_slicenextoutdim_larger!((T), RangeFrom<T>);
impl_slicenextoutdim_larger!((T), RangeTo<T>);
impl_slicenextoutdim_larger!((T), RangeToInclusive<T>);
impl_slicenextoutdim_larger!((), RangeFull);
impl_slicenextoutdim_larger!((), Slice);
impl_slicenextoutdim_larger!((), NewAxis);

/// Slice argument constructor.
///
/// `s![]` takes a list of ranges/slices/indices/new-axes, separated by comma,
/// with optional step sizes that are separated from the range by a semicolon.
/// It is converted into a [`&SliceInfo`] instance.
///
/// [`&SliceInfo`]: struct.SliceInfo.html
///
/// Each range/slice/index uses signed indices, where a negative value is
/// counted from the end of the axis. Step sizes are also signed and may be
/// negative, but must not be zero.
///
/// The syntax is `s![` *[ axis-slice-info [, axis-slice-info [ , ... ] ] ]*
/// `]`, where *axis-slice-info* is any of the following:
///
/// * *index*: an index to use for taking a subview with respect to that axis.
///   (The index is selected. The axis is removed except with
///   [`.slice_collapse()`].)
/// * *range*: a range with step size 1 to use for slicing that axis.
/// * *range* `;` *step*: a range with step size *step* to use for slicing that axis.
/// * *slice*: a [`Slice`] instance to use for slicing that axis.
/// * *slice* `;` *step*: a range constructed from the start and end of a [`Slice`]
///   instance, with new step size *step*, to use for slicing that axis.
/// * *new-axis*: a [`NewAxis`] instance that represents the creation of a new axis.
///
/// [`Slice`]: struct.Slice.html
/// [`NewAxis`]: struct.NewAxis.html
///
/// The number of *axis-slice-info*, not including *new-axis*, must match the
/// number of axes in the array. *index*, *range*, *slice*, *step*, and
/// *new-axis* can be expressions. *index* must be of type `isize`, `usize`, or
/// `i32`. *range* must be of type `Range<I>`, `RangeTo<I>`, `RangeFrom<I>`, or
/// `RangeFull` where `I` is `isize`, `usize`, or `i32`. *step* must be a type
/// that can be converted to `isize` with the `as` keyword.
///
/// For example `s![0..4;2, 6, 1..5, NewAxis]` is a slice of the first axis for
/// 0..4 with step size 2, a subview of the second axis at index 6, a slice of
/// the third axis for 1..5 with default step size 1, and a new axis of length
/// 1 at the end of the shape. The input array must have 3 dimensions. The
/// resulting slice would have shape `[2, 4, 1]` for [`.slice()`],
/// [`.slice_mut()`], and [`.slice_move()`], and shape `[2, 1, 4]` for
/// [`.slice_collapse()`].
///
/// [`.slice()`]: struct.ArrayBase.html#method.slice
/// [`.slice_mut()`]: struct.ArrayBase.html#method.slice_mut
/// [`.slice_move()`]: struct.ArrayBase.html#method.slice_move
/// [`.slice_collapse()`]: struct.ArrayBase.html#method.slice_collapse
///
/// See also [*Slicing*](struct.ArrayBase.html#slicing).
///
/// # Example
///
/// ```
/// use ndarray::{s, Array2, ArrayView2};
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
///
/// # Negative *step*
///
/// The behavior of negative *step* arguments is most easily understood with
/// slicing as a two-step process:
///
/// 1. First, perform a slice with *range*.
///
/// 2. If *step* is positive, start with the front of the slice; if *step* is
///    negative, start with the back of the slice. Then, add *step* until
///    reaching the other end of the slice (inclusive).
///
/// An equivalent way to think about step 2 is, "If *step* is negative, reverse
/// the slice. Start at the front of the (possibly reversed) slice, and add
/// *step.abs()* until reaching the back of the slice (inclusive)."
///
/// For example,
///
/// ```
/// # use ndarray::prelude::*;
/// #
/// # fn main() {
/// let arr = array![0, 1, 2, 3];
/// assert_eq!(arr.slice(s![1..3;-1]), array![2, 1]);
/// assert_eq!(arr.slice(s![1..;-2]), array![3, 1]);
/// assert_eq!(arr.slice(s![0..4;-2]), array![3, 1]);
/// assert_eq!(arr.slice(s![0..;-2]), array![3, 1]);
/// assert_eq!(arr.slice(s![..;-2]), array![3, 1]);
/// # }
/// ```
#[macro_export]
macro_rules! s(
    // convert a..b;c into @convert(a..b, c), final item
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr;$s:expr) => {
        match $r {
            r => {
                let in_dim = $crate::SliceNextInDim::next_dim(&r, $in_dim);
                let out_dim = $crate::SliceNextOutDim::next_dim(&r, $out_dim);
                #[allow(unsafe_code)]
                unsafe {
                    $crate::SliceInfo::new_unchecked(
                        [$($stack)* $crate::s!(@convert r, $s)],
                        in_dim,
                        out_dim,
                    )
                }
            }
        }
    };
    // convert a..b into @convert(a..b), final item
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr) => {
        match $r {
            r => {
                let in_dim = $crate::SliceNextInDim::next_dim(&r, $in_dim);
                let out_dim = $crate::SliceNextOutDim::next_dim(&r, $out_dim);
                #[allow(unsafe_code)]
                unsafe {
                    $crate::SliceInfo::new_unchecked(
                        [$($stack)* $crate::s!(@convert r)],
                        in_dim,
                        out_dim,
                    )
                }
            }
        }
    };
    // convert a..b;c into @convert(a..b, c), final item, trailing comma
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr;$s:expr ,) => {
        $crate::s![@parse $in_dim, $out_dim, [$($stack)*] $r;$s]
    };
    // convert a..b into @convert(a..b), final item, trailing comma
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr ,) => {
        $crate::s![@parse $in_dim, $out_dim, [$($stack)*] $r]
    };
    // convert a..b;c into @convert(a..b, c)
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr;$s:expr, $($t:tt)*) => {
        match $r {
            r => {
                $crate::s![@parse
                   $crate::SliceNextInDim::next_dim(&r, $in_dim),
                   $crate::SliceNextOutDim::next_dim(&r, $out_dim),
                   [$($stack)* $crate::s!(@convert r, $s),]
                   $($t)*
                ]
            }
        }
    };
    // convert a..b into @convert(a..b)
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr, $($t:tt)*) => {
        match $r {
            r => {
                $crate::s![@parse
                   $crate::SliceNextInDim::next_dim(&r, $in_dim),
                   $crate::SliceNextOutDim::next_dim(&r, $out_dim),
                   [$($stack)* $crate::s!(@convert r),]
                   $($t)*
                ]
            }
        }
    };
    // empty call, i.e. `s![]`
    (@parse ::std::marker::PhantomData::<$crate::Ix0>, ::std::marker::PhantomData::<$crate::Ix0>, []) => {
        {
            #[allow(unsafe_code)]
            unsafe {
                $crate::SliceInfo::new_unchecked(
                    [],
                    ::std::marker::PhantomData::<$crate::Ix0>,
                    ::std::marker::PhantomData::<$crate::Ix0>,
                )
            }
        }
    };
    // Catch-all clause for syntax errors
    (@parse $($t:tt)*) => { compile_error!("Invalid syntax in s![] call.") };
    // convert range/index/new-axis into AxisSliceInfo
    (@convert $r:expr) => {
        <$crate::AxisSliceInfo as ::std::convert::From<_>>::from($r)
    };
    // convert range/index/new-axis and step into AxisSliceInfo
    (@convert $r:expr, $s:expr) => {
        <$crate::AxisSliceInfo as ::std::convert::From<_>>::from($r).step_by($s as isize)
    };
    ($($t:tt)*) => {
        // The extra `*&` is a workaround for this compiler bug:
        // https://github.com/rust-lang/rust/issues/23014
        &*&$crate::s![@parse
              ::std::marker::PhantomData::<$crate::Ix0>,
              ::std::marker::PhantomData::<$crate::Ix0>,
              []
              $($t)*
        ]
    };
);

/// Slicing information describing multiple mutable, disjoint slices.
///
/// It's unfortunate that we need `'a` and `A` to be parameters of the trait,
/// but they're necessary until Rust supports generic associated types.
pub trait MultiSlice<'a, A, D>
where
    A: 'a,
    D: Dimension,
{
    /// The type of the slices created by `.multi_slice_move()`.
    type Output;

    /// Split the view into multiple disjoint slices.
    ///
    /// **Panics** if performing any individual slice panics or if the slices
    /// are not disjoint (i.e. if they intersect).
    fn multi_slice_move(&self, view: ArrayViewMut<'a, A, D>) -> Self::Output;
}

impl<'a, A, D> MultiSlice<'a, A, D> for ()
where
    A: 'a,
    D: Dimension,
{
    type Output = ();

    fn multi_slice_move(&self, _view: ArrayViewMut<'a, A, D>) -> Self::Output {}
}

impl<'a, A, D, I0> MultiSlice<'a, A, D> for (&I0,)
where
    A: 'a,
    D: Dimension,
    I0: CanSlice<D>,
{
    type Output = (ArrayViewMut<'a, A, I0::OutDim>,);

    fn multi_slice_move(&self, view: ArrayViewMut<'a, A, D>) -> Self::Output {
        (view.slice_move(self.0),)
    }
}

macro_rules! impl_multislice_tuple {
    ([$($but_last:ident)*] $last:ident) => {
        impl_multislice_tuple!(@def_impl ($($but_last,)* $last,), [$($but_last)*] $last);
    };
    (@def_impl ($($all:ident,)*), [$($but_last:ident)*] $last:ident) => {
        impl<'a, A, D, $($all,)*> MultiSlice<'a, A, D> for ($(&$all,)*)
        where
            A: 'a,
            D: Dimension,
            $($all: CanSlice<D>,)*
        {
            type Output = ($(ArrayViewMut<'a, A, $all::OutDim>,)*);

            fn multi_slice_move(&self, view: ArrayViewMut<'a, A, D>) -> Self::Output {
                #[allow(non_snake_case)]
                let &($($all,)*) = self;

                let shape = view.raw_dim();
                assert!(!impl_multislice_tuple!(@intersects_self &shape, ($($all,)*)));

                let raw_view = view.into_raw_view_mut();
                unsafe {
                    (
                        $(raw_view.clone().slice_move($but_last).deref_into_view_mut(),)*
                        raw_view.slice_move($last).deref_into_view_mut(),
                    )
                }
            }
        }
    };
    (@intersects_self $shape:expr, ($head:expr,)) => {
        false
    };
    (@intersects_self $shape:expr, ($head:expr, $($tail:expr,)*)) => {
        $(slices_intersect($shape, $head, $tail)) ||*
            || impl_multislice_tuple!(@intersects_self $shape, ($($tail,)*))
    };
}

impl_multislice_tuple!([I0] I1);
impl_multislice_tuple!([I0 I1] I2);
impl_multislice_tuple!([I0 I1 I2] I3);
impl_multislice_tuple!([I0 I1 I2 I3] I4);
impl_multislice_tuple!([I0 I1 I2 I3 I4] I5);

impl<'a, A, D, T> MultiSlice<'a, A, D> for &T
where
    A: 'a,
    D: Dimension,
    T: MultiSlice<'a, A, D>,
{
    type Output = T::Output;

    fn multi_slice_move(&self, view: ArrayViewMut<'a, A, D>) -> Self::Output {
        T::multi_slice_move(self, view)
    }
}
