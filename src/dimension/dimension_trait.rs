// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::fmt::Debug;
use std::ops::{Index, IndexMut};
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign};

use itertools::{enumerate, zip};

use {Ix, Ixs, Ix0, Ix1, Ix2, Ix3, IxDyn, Dim, Si};
use IntoDimension;
use RemoveAxis;
use {ArrayView1, ArrayViewMut1};
use {zipsl, zipsl_mut, ZipExt};
use Axis;
use super::{
    stride_offset,
    stride_offset_checked,
    DimPrivate,
};
use super::conversion::Convert;
use super::axes_of;

/// Array shape and index trait.
///
/// This trait defines a number of methods and operations that can be used on
/// dimensions and indices.
///
/// ***Note:*** *Don't implement this trait.*
pub unsafe trait Dimension : Clone + Eq + Debug + Send + Sync + Default +
    IndexMut<usize, Output=usize> +
    Add<Self, Output=Self> +
    AddAssign + for<'x> AddAssign<&'x Self> +
    Sub<Self, Output=Self> +
    SubAssign + for<'x> SubAssign<&'x Self> +
    Mul<usize, Output=Self> + Mul<Self, Output=Self> +
    MulAssign + for<'x> MulAssign<&'x Self> + MulAssign<usize>

{
    /// `SliceArg` is the type which is used to specify slicing for this
    /// dimension.
    ///
    /// For the fixed size dimensions it is a fixed size array of the correct
    /// size, which you pass by reference. For the `Vec` dimension it is
    /// a slice.
    ///
    /// - For `Ix1`: `[Si; 1]`
    /// - For `Ix2`: `[Si; 2]`
    /// - and so on..
    /// - For `Vec<Ix>`: `[Si]`
    ///
    /// The easiest way to create a `&SliceArg` is using the macro
    /// [`s![]`](macro.s!.html).
    type SliceArg: ?Sized + AsRef<[Si]>;
    /// Pattern matching friendly form of the dimension value.
    ///
    /// - For `Ix1`: `usize`,
    /// - For `Ix2`: `(usize, usize)`
    /// - and so on..
    /// - For `Vec<Ix>`: `Vec<usize>`,
    type Pattern: IntoDimension<Dim=Self>;
    // Next smaller dimension (if it exists)
    #[doc(hidden)]
    type TrySmaller: Dimension;
    #[doc(hidden)]
    fn ndim(&self) -> usize;

    /// Convert the dimension into a pattern matching friendly value.
    fn into_pattern(self) -> Self::Pattern;

    /// Compute the size of the dimension (number of elements)
    fn size(&self) -> usize {
        self.slice().iter().fold(1, |s, &a| s * a as usize)
    }

    /// Compute the size while checking for overflow.
    fn size_checked(&self) -> Option<usize> {
        self.slice().iter().fold(Some(1), |s, &a| s.and_then(|s_| s_.checked_mul(a)))
    }

    #[doc(hidden)]
    fn slice(&self) -> &[Ix];

    #[doc(hidden)]
    fn slice_mut(&mut self) -> &mut [Ix];

    /// Borrow as a read-only array view.
    fn as_array_view(&self) -> ArrayView1<Ix> {
        ArrayView1::from(self.slice())
    }

    /// Borrow as a read-write array view.
    fn as_array_view_mut(&mut self) -> ArrayViewMut1<Ix> {
        ArrayViewMut1::from(self.slice_mut())
    }

    #[doc(hidden)]
    fn equal(&self, rhs: &Self) -> bool {
        self.slice() == rhs.slice()
    }

    #[doc(hidden)]
    fn default_strides(&self) -> Self {
        // Compute default array strides
        // Shape (a, b, c) => Give strides (b * c, c, 1)
        let mut strides = self.clone();
        {
            let mut it = strides.slice_mut().iter_mut().rev();
            // Set first element to 1
            while let Some(rs) = it.next() {
                *rs = 1;
                break;
            }
            let mut cum_prod = 1;
            for (rs, dim) in it.zip(self.slice().iter().rev()) {
                cum_prod *= *dim;
                *rs = cum_prod;
            }
        }
        strides
    }

    #[doc(hidden)]
    fn fortran_strides(&self) -> Self {
        // Compute fortran array strides
        // Shape (a, b, c) => Give strides (1, a, a * b)
        let mut strides = self.clone();
        {
            let mut it = strides.slice_mut().iter_mut();
            // Set first element to 1
            while let Some(rs) = it.next() {
                *rs = 1;
                break;
            }
            let mut cum_prod = 1;
            for (rs, dim) in it.zip(self.slice()) {
                cum_prod *= *dim;
                *rs = cum_prod;
            }
        }
        strides
    }

    #[doc(hidden)]
    #[inline]
    fn first_index(&self) -> Option<Self> {
        for ax in self.slice().iter() {
            if *ax == 0 {
                return None;
            }
        }
        let mut index = self.clone();
        for rr in index.slice_mut().iter_mut() {
            *rr = 0;
        }
        Some(index)
    }

    #[doc(hidden)]
    /// Iteration -- Use self as size, and return next index after `index`
    /// or None if there are no more.
    // FIXME: use &Self for index or even &mut?
    #[inline]
    fn next_for(&self, index: Self) -> Option<Self> {
        let mut index = index;
        let mut done = false;
        for (&dim, ix) in zip(self.slice(), index.slice_mut()).rev() {
            *ix += 1;
            if *ix == dim {
                *ix = 0;
            } else {
                done = true;
                break;
            }
        }
        if done {
            Some(index)
        } else {
            None
        }
    }

    #[doc(hidden)]
    /// Return stride offset for index.
    fn stride_offset(index: &Self, strides: &Self) -> isize {
        let mut offset = 0;
        for (&i, &s) in zipsl(index.slice(), strides.slice()) {
            offset += stride_offset(i, s);
        }
        offset
    }

    #[doc(hidden)]
    /// Return stride offset for this dimension and index.
    fn stride_offset_checked(&self, strides: &Self, index: &Self) -> Option<isize> {
        stride_offset_checked(self.slice(), strides.slice(), index.slice())
    }

    #[doc(hidden)]
    fn last_elem(&self) -> usize {
        if self.ndim() == 0 { 0 } else { self.slice()[self.ndim() - 1] }
    }

    #[doc(hidden)]
    fn set_last_elem(&mut self, i: usize) {
        let nd = self.ndim();
        self.slice_mut()[nd - 1] = i;
    }

    #[doc(hidden)]
    /// Modify dimension, strides and return data pointer offset
    ///
    /// **Panics** if `slices` does not correspond to the number of axes,
    /// if any stride is 0, or if any index is out of bounds.
    fn do_slices(dim: &mut Self, strides: &mut Self, slices: &Self::SliceArg) -> isize {
        let slices = slices.as_ref();
        let mut offset = 0;
        ndassert!(slices.len() == dim.slice().len(),
                  "SliceArg {:?}'s length does not match dimension {:?}",
                  slices, dim);
        for (dr, sr, &slc) in zipsl_mut(dim.slice_mut(), strides.slice_mut()).zip_cons(slices)
        {
            let m = *dr;
            let mi = m as Ixs;
            let Si(b1, opt_e1, s1) = slc;
            let e1 = opt_e1.unwrap_or(mi);

            let b1 = abs_index(mi, b1);
            let mut e1 = abs_index(mi, e1);
            if e1 < b1 { e1 = b1; }

            ndassert!(b1 <= m,
                      concat!("Slice begin {} is past end of axis of length {}",
                              " (for SliceArg {:?})"),
                      b1, m, slices);
            ndassert!(e1 <= m,
                      concat!("Slice end {} is past end of axis of length {}",
                              " (for SliceArg {:?})"),
                      e1, m, slices);

            let m = e1 - b1;
            // stride
            let s = (*sr) as Ixs;

            // Data pointer offset
            offset += stride_offset(b1, *sr);
            // Adjust for strides
            ndassert!(s1 != 0,
                      concat!("Slice stride must not be none", 
                              "(for SliceArg {:?})"),
                      slices);
            // How to implement negative strides:
            //
            // Increase start pointer by
            // old stride * (old dim - 1)
            // to put the pointer completely in the other end
            if s1 < 0 {
                offset += stride_offset(m - 1, *sr);
            }

            let s_prim = s * s1;

            let d = m / s1.abs() as Ix;
            let r = m % s1.abs() as Ix;
            let m_prim = d + if r > 0 { 1 } else { 0 };

            // Update dimension and stride coordinate
            *dr = m_prim;
            *sr = s_prim as Ix;
        }
        offset
    }

    #[doc(hidden)]
    fn is_contiguous(dim: &Self, strides: &Self) -> bool {
        let defaults = dim.default_strides();
        if strides.equal(&defaults) {
            return true;
        }
        if dim.ndim() == 1 { return false; }
        let order = strides._fastest_varying_stride_order();
        let strides = strides.slice();

        // FIXME: Negative strides
        let dim_slice = dim.slice();
        let mut cstride = 1;
        for &i in order.slice() {
            // a dimension of length 1 can have unequal strides
            if dim_slice[i] != 1 && strides[i] != cstride {
                return false;
            }
            cstride *= dim_slice[i];
        }
        true
    }

    /// Return the axis ordering corresponding to the fastest variation
    /// (in ascending order).
    ///
    /// Assumes that no stride value appears twice. This cannot yield the correct
    /// result the strides are not positive.
    #[doc(hidden)]
    fn _fastest_varying_stride_order(&self) -> Self {
        let mut indices = self.clone();
        for (i, elt) in enumerate(indices.slice_mut()) {
            *elt = i;
        }
        let strides = self.slice();
        indices.slice_mut().sort_by_key(|&i| strides[i]);
        indices
    }

    /// Compute the minimum stride axis (absolute value), under the constraint
    /// that the length of the axis is > 1;
    #[doc(hidden)]
    fn min_stride_axis(&self, strides: &Self) -> Axis {
        let n = match self.ndim() {
            0 => panic!("min_stride_axis: Array must have ndim > 0"),
            1 => return Axis(0),
            n => n,
        };
        axes_of(self, strides)
            .rev()
            .min_by_key(|ax| ax.stride().abs())
            .map_or(Axis(n - 1), |ax| ax.axis())
    }

    /// Compute the maximum stride axis (absolute value), under the constraint
    /// that the length of the axis is > 1;
    #[doc(hidden)]
    fn max_stride_axis(&self, strides: &Self) -> Axis {
        match self.ndim() {
            0 => panic!("max_stride_axis: Array must have ndim > 0"),
            1 => return Axis(0),
            _ => { }
        }
        axes_of(self, strides)
            .filter(|ax| ax.len() > 1)
            .max_by_key(|ax| ax.stride().abs())
            .map_or(Axis(0), |ax| ax.axis())
    }

    #[doc(hidden)]
    fn try_remove_axis(&self, axis: Axis) -> Self::TrySmaller;
}

// utility functions

#[inline]
fn abs_index(len: Ixs, index: Ixs) -> Ix {
    if index < 0 {
        (len + index) as Ix
    } else {
        index as Ix
    }
}


// Dimension impls


unsafe impl Dimension for Dim<[Ix; 0]> {
    type SliceArg = [Si; 0];
    type Pattern = ();
    type TrySmaller = Self;
    // empty product is 1 -> size is 1
    #[inline]
    fn ndim(&self) -> usize { 0 }
    #[inline]
    fn slice(&self) -> &[Ix] { &[] }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix] { &mut [] }
    #[inline]
    fn _fastest_varying_stride_order(&self) -> Self { Ix0() }
    #[inline]
    fn into_pattern(self) -> Self::Pattern { }
    #[inline]
    fn next_for(&self, _index: Self) -> Option<Self> {
        None
    }
    #[inline]
    fn try_remove_axis(&self, _ignore: Axis) -> Self::TrySmaller {
        *self
    }
}


unsafe impl Dimension for Dim<[Ix; 1]> {
    type SliceArg = [Si; 1];
    type Pattern = Ix;
    type TrySmaller = <Self as RemoveAxis>::Smaller;
    #[inline]
    fn ndim(&self) -> usize { 1 }
    #[inline]
    fn slice(&self) -> &[Ix] { self.ix() }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix] { self.ixm() }
    #[inline]
    fn into_pattern(self) -> Self::Pattern {
        get!(&self, 0)
    }
    #[inline]
    fn next_for(&self, mut index: Self) -> Option<Self> {
        getm!(index, 0) += 1;
        if get!(&index, 0) < get!(self, 0) {
            Some(index)
        } else {
            None
        }
    }

    #[inline]
    fn equal(&self, rhs: &Self) -> bool {
        get!(self, 0) == get!(rhs, 0)
    }

    #[inline]
    fn size(&self) -> usize { get!(self, 0) }
    #[inline]
    fn size_checked(&self) -> Option<usize> { Some(get!(self, 0)) }

    #[inline]
    fn default_strides(&self) -> Self {
        Ix1(1)
    }

    #[inline]
    fn _fastest_varying_stride_order(&self) -> Self {
        Ix1(0)
    }

    #[inline(always)]
    fn min_stride_axis(&self, _: &Self) -> Axis {
        Axis(0)
    }

    #[inline(always)]
    fn max_stride_axis(&self, _: &Self) -> Axis {
        Axis(0)
    }

    #[inline]
    fn first_index(&self) -> Option<Self> {
        if get!(self, 0) != 0 {
            Some(Ix1(0))
        } else {
            None
        }
    }

    /// Self is an index, return the stride offset
    #[inline(always)]
    fn stride_offset(index: &Self, stride: &Self) -> isize {
        stride_offset(get!(index, 0), get!(stride, 0))
    }

    /// Return stride offset for this dimension and index.
    #[inline]
    fn stride_offset_checked(&self, stride: &Self, index: &Self) -> Option<isize> {
        if get!(index, 0) < get!(self, 0) {
            Some(stride_offset(get!(index, 0), get!(stride, 0)))
        } else {
            None
        }
    }
    #[inline]
    fn try_remove_axis(&self, axis: Axis) -> Self::TrySmaller {
        self.remove_axis(axis)
    }
}

unsafe impl Dimension for Dim<[Ix; 2]> {
    type SliceArg = [Si; 2];
    type Pattern = (Ix, Ix);
    type TrySmaller = <Self as RemoveAxis>::Smaller;
    #[inline]
    fn ndim(&self) -> usize { 2 }
    #[inline]
    fn into_pattern(self) -> Self::Pattern {
        self.ix().convert()
    }
    #[inline]
    fn slice(&self) -> &[Ix] { self.ix() }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix] { self.ixm() }
    #[inline]
    fn next_for(&self, index: Self) -> Option<Self> {
        let mut i = get!(&index, 0);
        let mut j = get!(&index, 1);
        let imax = get!(self, 0);
        let jmax = get!(self, 1);
        j += 1;
        if j >= jmax {
            j = 0;
            i += 1;
            if i >= imax {
                return None;
            }
        }
        Some(Ix2(i, j))
    }

    #[inline]
    fn equal(&self, rhs: &Self) -> bool {
        get!(self, 0) == get!(rhs, 0) && get!(self, 1) == get!(rhs, 1)
    }

    #[inline]
    fn size(&self) -> usize { get!(self, 0) * get!(self, 1) }

    #[inline]
    fn size_checked(&self) -> Option<usize> {
        let m = get!(self, 0);
        let n = get!(self, 1);
        (m as usize).checked_mul(n as usize)
    }

    #[inline]
    fn last_elem(&self) -> usize {
        get!(self, 1)
    }

    #[inline]
    fn set_last_elem(&mut self, i: usize) {
        getm!(self, 1) = i;
    }

    #[inline]
    fn default_strides(&self) -> Self {
        // Compute default array strides
        // Shape (a, b, c) => Give strides (b * c, c, 1)
        Ix2(get!(self, 1), 1)
    }
    #[inline]
    fn fortran_strides(&self) -> Self {
        Ix2(1, get!(self, 0))
    }

    #[inline]
    fn _fastest_varying_stride_order(&self) -> Self {
        if get!(self, 0) as Ixs <= get!(self, 1) as Ixs { Ix2(0, 1) } else { Ix2(1, 0) }
    }

    #[inline]
    fn min_stride_axis(&self, strides: &Self) -> Axis {
        let s = get!(strides, 0) as Ixs;
        let t = get!(strides, 1) as Ixs;
        if s.abs() < t.abs() {
            Axis(0)
        } else {
            Axis(1)
        }
    }
    
    #[inline]
    fn first_index(&self) -> Option<Self> {
        let m = get!(self, 0);
        let n = get!(self, 1);
        if m != 0 && n != 0 {
            Some(Ix2(0, 0))
        } else {
            None
        }
    }

    /// Self is an index, return the stride offset
    #[inline(always)]
    fn stride_offset(index: &Self, strides: &Self) -> isize {
        let i = get!(index, 0);
        let j = get!(index, 1);
        let s = get!(strides, 0);
        let t = get!(strides, 1);
        stride_offset(i, s) + stride_offset(j, t)
    }

    /// Return stride offset for this dimension and index.
    #[inline]
    fn stride_offset_checked(&self, strides: &Self, index: &Self) -> Option<isize>
    {
        let m = get!(self, 0);
        let n = get!(self, 1);
        let i = get!(index, 0);
        let j = get!(index, 1);
        let s = get!(strides, 0);
        let t = get!(strides, 1);
        if i < m && j < n {
            Some(stride_offset(i, s) + stride_offset(j, t))
        } else {
            None
        }
    }
    #[inline]
    fn try_remove_axis(&self, axis: Axis) -> Self::TrySmaller {
        self.remove_axis(axis)
    }
}

unsafe impl Dimension for Dim<[Ix; 3]> {
    type SliceArg = [Si; 3];
    type Pattern = (Ix, Ix, Ix);
    type TrySmaller = <Self as RemoveAxis>::Smaller;
    #[inline]
    fn ndim(&self) -> usize { 3 }
    #[inline]
    fn into_pattern(self) -> Self::Pattern {
        self.ix().convert()
    }
    #[inline]
    fn slice(&self) -> &[Ix] { self.ix() }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix] { self.ixm() }

    #[inline]
    fn size(&self) -> usize {
        let m = get!(self, 0);
        let n = get!(self, 1);
        let o = get!(self, 2);
        m as usize * n as usize * o as usize
    }

    #[inline]
    fn next_for(&self, index: Self) -> Option<Self> {
        let mut i = get!(&index, 0);
        let mut j = get!(&index, 1);
        let mut k = get!(&index, 2);
        let imax = get!(self, 0);
        let jmax = get!(self, 1);
        let kmax = get!(self, 2);
        k += 1;
        if k == kmax {
            k = 0;
            j += 1;
            if j == jmax {
                j = 0;
                i += 1;
                if i == imax {
                    return None;
                }
            }
        }
        Some(Ix3(i, j, k))
    }

    /// Self is an index, return the stride offset
    #[inline]
    fn stride_offset(index: &Self, strides: &Self) -> isize {
        let i = get!(index, 0);
        let j = get!(index, 1);
        let k = get!(index, 2);
        let s = get!(strides, 0);
        let t = get!(strides, 1);
        let u = get!(strides, 2);
        stride_offset(i, s) + stride_offset(j, t) + stride_offset(k, u)
    }

    #[inline]
    fn _fastest_varying_stride_order(&self) -> Self {
        let mut stride = *self;
        let mut order = Ix3(0, 1, 2);
        macro_rules! swap {
            ($stride:expr, $order:expr, $x:expr, $y:expr) => {
                if $stride[$x] > $stride[$y] {
                    $stride.swap($x, $y);
                    $order.ixm().swap($x, $y);
                }
            }
        }
        {
            // stable sorting network for 3 elements
            let strides = stride.slice_mut();
            swap![strides, order, 1, 2];
            swap![strides, order, 0, 1];
            swap![strides, order, 1, 2];
        }
        order
    }
    #[inline]
    fn try_remove_axis(&self, axis: Axis) -> Self::TrySmaller {
        self.remove_axis(axis)
    }
}

macro_rules! large_dim {
    ($n:expr, $name:ident, $($ix:ident),+) => (
        unsafe impl Dimension for Dim<[Ix; $n]> {
            type SliceArg = [Si; $n];
            type Pattern = ($($ix,)*);
            type TrySmaller = <Self as RemoveAxis>::Smaller;
            #[inline]
            fn ndim(&self) -> usize { $n }
            #[inline]
            fn into_pattern(self) -> Self::Pattern {
                self.ix().convert()
            }
            #[inline]
            fn slice(&self) -> &[Ix] { self.ix() }
            #[inline]
            fn slice_mut(&mut self) -> &mut [Ix] { self.ixm() }
            #[inline]
            fn try_remove_axis(&self, axis: Axis) -> Self::TrySmaller {
                self.remove_axis(axis)
            }
        }
    )
}

large_dim!(4, Ix4, Ix, Ix, Ix, Ix);
large_dim!(5, Ix5, Ix, Ix, Ix, Ix, Ix);
large_dim!(6, Ix6, Ix, Ix, Ix, Ix, Ix, Ix);

/// Vec<Ix> is a "dynamic" index, pretty hard to use when indexing,
/// and memory wasteful, but it allows an arbitrary and dynamic number of axes.
unsafe impl Dimension for IxDyn
{
    type SliceArg = [Si];
    type Pattern = Self;
    type TrySmaller = <Self as RemoveAxis>::Smaller;
    #[inline]
    fn ndim(&self) -> usize { self.ix().len() }
    #[inline]
    fn slice(&self) -> &[Ix] { self.ix() }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix] { self.ixm() }
    #[inline]
    fn into_pattern(self) -> Self::Pattern {
        self
    }
    #[inline]
    fn try_remove_axis(&self, axis: Axis) -> Self::TrySmaller {
        if self.ndim() > 0 {
            self.remove_axis(axis)
        } else {
            self.clone()
        }
    }
}

impl<J> Index<J> for Dim<Vec<usize>>
    where Vec<usize>: Index<J>,
{
    type Output = <Vec<usize> as Index<J>>::Output;
    fn index(&self, index: J) -> &Self::Output {
        &self.ix()[index]
    }
}

impl<J> IndexMut<J> for Dim<Vec<usize>>
    where Vec<usize>: IndexMut<J>,
{
    fn index_mut(&mut self, index: J) -> &mut Self::Output {
        &mut self.ixm()[index]
    }
}
