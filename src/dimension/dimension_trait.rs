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

use {Ix, Ixs, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, Dim, Si};
use IntoDimension;
use {ArrayView1, ArrayViewMut1};
use {zipsl, zipsl_mut, ZipExt};
use super::{
    stride_offset,
    DimPrivate,
};
use super::conversion::Convert;

/// Array shape and index trait.
///
/// `unsafe` because of the assumptions in the default methods.
///
/// ***Don't implement this trait, it will evolve at will.***
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
            for rs in it.by_ref() {
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
            for rs in it.by_ref() {
                *rs = 1;
                break;
            }
            let mut cum_prod = 1;
            for (rs, dim) in it.zip(self.slice().iter()) {
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
        let mut offset = 0;
        for (&d, &i, &s) in zipsl(self.slice(), index.slice()).zip_cons(strides.slice())
        {
            if i >= d {
                return None;
            }
            offset += stride_offset(i, s);
        }
        Some(offset)
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
        assert!(slices.len() == dim.slice().len());
        for (dr, sr, &slc) in zipsl_mut(dim.slice_mut(), strides.slice_mut()).zip_cons(slices)
        {
            let m = *dr;
            let mi = m as Ixs;
            let Si(b1, opt_e1, s1) = slc;
            let e1 = opt_e1.unwrap_or(mi);

            let b1 = abs_index(mi, b1);
            let mut e1 = abs_index(mi, e1);
            if e1 < b1 { e1 = b1; }

            assert!(b1 <= m);
            assert!(e1 <= m);

            let m = e1 - b1;
            // stride
            let s = (*sr) as Ixs;

            // Data pointer offset
            offset += stride_offset(b1, *sr);
            // Adjust for strides
            assert!(s1 != 0);
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


unsafe impl Dimension for Ix0 {
    type SliceArg = [Si; 0];
    type Pattern = ();
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
}


unsafe impl Dimension for Ix1 {
    type SliceArg = [Si; 1];
    type Pattern = Ix;
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
}

unsafe impl Dimension for Ix2 {
    type SliceArg = [Si; 2];
    type Pattern = (Ix, Ix);
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
}

unsafe impl Dimension for Ix3 {
    type SliceArg = [Si; 3];
    type Pattern = (Ix, Ix, Ix);
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
}

macro_rules! large_dim {
    ($n:expr, $name:ident, $($ix:ident),+) => (
        unsafe impl Dimension for $name {
            type SliceArg = [Si; $n];
            type Pattern = ($($ix,)*);
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
    fn ndim(&self) -> usize { self.ix().len() }
    fn slice(&self) -> &[Ix] { self.ix() }
    fn slice_mut(&mut self) -> &mut [Ix] { self.ixm() }
    #[inline]
    fn into_pattern(self) -> Self::Pattern {
        self
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
