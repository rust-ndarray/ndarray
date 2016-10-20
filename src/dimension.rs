// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use std::cmp::Ordering;
use std::fmt::Debug;
use std::slice;
use itertools::{enumerate, zip};

use super::{Si, Ix, Ixs};
use super::{zipsl, zipsl_mut};
use error::{from_kind, ErrorKind, ShapeError};
use ZipExt;

/// Calculate offset from `Ix` stride converting sign properly
#[inline]
pub fn stride_offset(n: Ix, stride: Ix) -> isize {
    (n as isize) * ((stride as Ixs) as isize)
}

/// Check whether the given `dim` and `stride` lead to overlapping indices
///
/// There is overlap if, when iterating through the dimensions in the order
/// of maximum variation, the current stride is inferior to the sum of all
/// preceding strides multiplied by their corresponding dimensions.
///
/// The current implementation assumes strides to be positive
pub fn dim_stride_overlap<D: Dimension>(dim: &D, strides: &D) -> bool {
    let order = strides._fastest_varying_stride_order();

    let dim = dim.slice();
    let strides = strides.slice();
    let mut prev_offset = 1;
    for &index in order.slice() {
        let d = dim[index];
        let s = strides[index];
        // any stride is ok if dimension is 1
        if d != 1 && (s as isize) < prev_offset {
            return true;
        }
        prev_offset = stride_offset(d, s);
    }
    false
}

/// Check whether the given dimension and strides are memory safe
/// to index the provided slice.
///
/// To be safe, no stride may be negative, and the offset corresponding
/// to the last element of each dimension should be smaller than the length
/// of the slice. Also, the strides should not allow a same element to be
/// referenced by two different index.
pub fn can_index_slice<A, D: Dimension>(data: &[A], dim: &D, strides: &D)
    -> Result<(), ShapeError>
{
    // check lengths of axes.
    let len = match dim.size_checked() {
        Some(l) => l,
        None => return Err(from_kind(ErrorKind::OutOfBounds)),
    };
    // check if strides are strictly positive (zero ok for len 0)
    for &s in strides.slice() {
        let s = s as Ixs;
        if s < 1 && (len != 0 || s < 0) {
            return Err(from_kind(ErrorKind::Unsupported));
        }
    }
    if len == 0 {
        return Ok(());
    }
    // check that the maximum index is in bounds
    let mut last_index = dim.clone();
    for mut index in last_index.slice_mut().iter_mut() {
        *index -= 1;
    }
    if let Some(offset) = stride_offset_checked_arithmetic(dim,
                                                           strides,
                                                           &last_index)
    {
        // offset is guaranteed to be positive so no issue converting
        // to usize here
        if (offset as usize) >= data.len() {
            return Err(from_kind(ErrorKind::OutOfBounds));
        }
        if dim_stride_overlap(dim, strides) {
            return Err(from_kind(ErrorKind::Unsupported));
        }
    } else {
        return Err(from_kind(ErrorKind::OutOfBounds));
    }
    Ok(())
}

/// Return stride offset for this dimension and index.
///
/// Return None if the indices are out of bounds, or the calculation would wrap
/// around.
fn stride_offset_checked_arithmetic<D>(dim: &D, strides: &D, index: &D)
    -> Option<isize>
    where D: Dimension
{
    let mut offset = 0;
    for (&d, &i, &s) in zipsl(dim.slice(), index.slice()).zip_cons(strides.slice()) {
        if i >= d {
            return None;
        }

        if let Some(offset_) = (i as isize)
                                   .checked_mul((s as Ixs) as isize)
                                   .and_then(|x| x.checked_add(offset)) {
            offset = offset_;
        } else {
            return None;
        }
    }
    Some(offset)
}

/// Array shape and index trait.
///
/// `unsafe` because of the assumptions in the default methods.
///
/// ***Don't implement or call methods in this trait, its interface is internal
/// to the crate and will evolve at will.***
pub unsafe trait Dimension : Clone + Eq + Debug + Send + Sync + Default {
    /// `SliceArg` is the type which is used to specify slicing for this
    /// dimension.
    ///
    /// For the fixed size dimensions (tuples) it is a fixed size array
    /// of the correct size, which you pass by reference. For the `Vec`
    /// dimension it is a slice.
    ///
    /// - For `Ix`: `[Si; 1]`
    /// - For `(Ix, Ix)`: `[Si; 2]`
    /// - and so on..
    /// - For `Vec<Ix>`: `[Si]`
    ///
    /// The easiest way to create a `&SliceArg` is using the macro
    /// [`s![]`](macro.s!.html).
    type SliceArg: ?Sized + AsRef<[Si]>;
    #[doc(hidden)]
    fn ndim(&self) -> usize;
    #[doc(hidden)]
    fn slice(&self) -> &[Ix] {
        unsafe {
            slice::from_raw_parts(self as *const _ as *const Ix, self.ndim())
        }
    }

    #[doc(hidden)]
    fn slice_mut(&mut self) -> &mut [Ix] {
        unsafe {
            slice::from_raw_parts_mut(self as *mut _ as *mut Ix, self.ndim())
        }
    }

    #[doc(hidden)]
    fn size(&self) -> usize {
        self.slice().iter().fold(1, |s, &a| s * a as usize)
    }

    #[doc(hidden)]
    /// Compute the size while checking for overflow
    fn size_checked(&self) -> Option<usize> {
        self.slice().iter().fold(Some(1), |s, &a| s.and_then(|s_| s_.checked_mul(a)))
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

/// Implementation-specific extensions to `Dimension`
pub trait DimensionExt {
// note: many extensions go in the main trait if they need to be special-
// cased per dimension
    /// Get the dimension at `axis`.
    ///
    /// *Panics* if `axis` is out of bounds.
    #[inline]
    fn axis(&self, axis: Axis) -> Ix;

    /// Set the dimension at `axis`.
    ///
    /// *Panics* if `axis` is out of bounds.
    #[inline]
    fn set_axis(&mut self, axis: Axis, value: Ix);
}

impl<D> DimensionExt for D
    where D: Dimension
{
    #[inline]
    fn axis(&self, axis: Axis) -> Ix {
        self.slice()[axis.axis()]
    }

    #[inline]
    fn set_axis(&mut self, axis: Axis, value: Ix) {
        self.slice_mut()[axis.axis()] = value;
    }
}

impl<'a> DimensionExt for [Ix]
{
    #[inline]
    fn axis(&self, axis: Axis) -> Ix {
        self[axis.axis()]
    }

    #[inline]
    fn set_axis(&mut self, axis: Axis, value: Ix) {
        self[axis.axis()] = value;
    }
}

#[inline]
fn abs_index(len: Ixs, index: Ixs) -> Ix {
    if index < 0 {
        (len + index) as Ix
    } else {
        index as Ix
    }
}

/// Collapse axis `axis` and shift so that only subarray `index` is
/// available.
///
/// **Panics** if `index` is larger than the size of the axis
// FIXME: Move to Dimension trait
pub fn do_sub<A, D: Dimension>(dims: &mut D, ptr: &mut *mut A, strides: &D,
                               axis: usize, index: Ix) {
    let dim = dims.slice()[axis];
    let stride = strides.slice()[axis];
    assert!(index < dim);
    dims.slice_mut()[axis] = 1;
    let off = stride_offset(index, stride);
    unsafe {
        *ptr = ptr.offset(off);
    }
}


unsafe impl Dimension for () {
    type SliceArg = [Si; 0];
    // empty product is 1 -> size is 1
    #[inline]
    fn ndim(&self) -> usize { 0 }
    fn slice(&self) -> &[Ix] { &[] }
    fn slice_mut(&mut self) -> &mut [Ix] { &mut [] }
}

unsafe impl Dimension for Ix {
    type SliceArg = [Si; 1];
    #[inline]
    fn ndim(&self) -> usize { 1 }
    #[inline]
    fn size(&self) -> usize { *self as usize }
    #[inline]
    fn size_checked(&self) -> Option<usize> { Some(*self as usize) }

    #[inline]
    fn default_strides(&self) -> Self { 1 }

    #[inline]
    fn first_index(&self) -> Option<Ix> {
        if *self != 0 {
            Some(0)
        } else {
            None
        }
    }
    #[inline]
    fn next_for(&self, mut index: Ix) -> Option<Ix> {
        index += 1;
        if index < *self {
            Some(index)
        } else {
            None
        }
    }

    /// Self is an index, return the stride offset
    #[inline]
    fn stride_offset(index: &Ix, stride: &Ix) -> isize {
        stride_offset(*index, *stride)
    }

    /// Return stride offset for this dimension and index.
    #[inline]
    fn stride_offset_checked(&self, stride: &Ix, index: &Ix) -> Option<isize> {
        if *index < *self {
            Some(stride_offset(*index, *stride))
        } else {
            None
        }
    }
}

unsafe impl Dimension for (Ix, Ix) {
    type SliceArg = [Si; 2];
    #[inline]
    fn ndim(&self) -> usize { 2 }

    #[inline]
    fn size(&self) -> usize { let (m, n) = *self; m as usize * n as usize }

    #[inline]
    fn size_checked(&self) -> Option<usize> {
        let (m, n) = *self;
        (m as usize).checked_mul(n as usize)
    }

    #[inline]
    fn default_strides(&self) -> Self {
        // Compute default array strides
        // Shape (a, b, c) => Give strides (b * c, c, 1)
        (self.1, 1)
    }

    #[inline]
    fn _fastest_varying_stride_order(&self) -> Self {
        if self.0 as Ixs <= self.1 as Ixs { (0, 1) } else { (1, 0) }
    }

    #[inline]
    fn first_index(&self) -> Option<(Ix, Ix)> {
        let (m, n) = *self;
        if m != 0 && n != 0 {
            Some((0, 0))
        } else {
            None
        }
    }
    #[inline]
    fn next_for(&self, index: (Ix, Ix)) -> Option<(Ix, Ix)> {
        let (mut i, mut j) = index;
        let (imax, jmax) = *self;
        j += 1;
        if j == jmax {
            j = 0;
            i += 1;
            if i == imax {
                return None;
            }
        }
        Some((i, j))
    }

    /// Self is an index, return the stride offset
    #[inline]
    fn stride_offset(index: &(Ix, Ix), strides: &(Ix, Ix)) -> isize {
        let (i, j) = *index;
        let (s, t) = *strides;
        stride_offset(i, s) + stride_offset(j, t)
    }

    /// Return stride offset for this dimension and index.
    #[inline]
    fn stride_offset_checked(&self, strides: &(Ix, Ix), index: &(Ix, Ix)) -> Option<isize>
    {
        let (m, n) = *self;
        let (i, j) = *index;
        let (s, t) = *strides;
        if i < m && j < n {
            Some(stride_offset(i, s) + stride_offset(j, t))
        } else {
            None
        }
    }
}

unsafe impl Dimension for (Ix, Ix, Ix) {
    type SliceArg = [Si; 3];
    #[inline]
    fn ndim(&self) -> usize { 3 }
    #[inline]
    fn size(&self) -> usize { let (m, n, o) = *self; m as usize * n as usize * o as usize }
    #[inline]
    fn next_for(&self, index: (Ix, Ix, Ix)) -> Option<(Ix, Ix, Ix)> {
        let (mut i, mut j, mut k) = index;
        let (imax, jmax, kmax) = *self;
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
        Some((i, j, k))
    }

    /// Self is an index, return the stride offset
    #[inline]
    fn stride_offset(index: &(Ix, Ix, Ix), strides: &(Ix, Ix, Ix)) -> isize {
        let (i, j, k) = *index;
        let (s, t, u) = *strides;
        stride_offset(i, s) + stride_offset(j, t) + stride_offset(k, u)
    }

    #[inline]
    fn _fastest_varying_stride_order(&self) -> Self {
        let mut stride = *self;
        let mut order = (0, 1, 2);
        macro_rules! swap {
            ($stride:expr, $order:expr, $x:expr, $y:expr) => {
                if $stride[$x] > $stride[$y] {
                    $stride.swap($x, $y);
                    $order.swap($x, $y);
                }
            }
        }
        {
            // stable sorting network for 3 elements
            let order = order.slice_mut();
            let strides = stride.slice_mut();
            swap![strides, order, 1, 2];
            swap![strides, order, 0, 1];
            swap![strides, order, 1, 2];
        }
        order
    }
}

macro_rules! large_dim {
    ($n:expr, $($ix:ident),+) => (
        unsafe impl Dimension for ($($ix),+) {
            type SliceArg = [Si; $n];
            #[inline]
            fn ndim(&self) -> usize { $n }
        }
    )
}

large_dim!(4, Ix, Ix, Ix, Ix);
large_dim!(5, Ix, Ix, Ix, Ix, Ix);
large_dim!(6, Ix, Ix, Ix, Ix, Ix, Ix);
large_dim!(7, Ix, Ix, Ix, Ix, Ix, Ix, Ix);
large_dim!(8, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix);
large_dim!(9, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix);
large_dim!(10, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix);
large_dim!(11, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix);
large_dim!(12, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix);

/// Vec<Ix> is a "dynamic" index, pretty hard to use when indexing,
/// and memory wasteful, but it allows an arbitrary and dynamic number of axes.
unsafe impl Dimension for Vec<Ix>
{
    type SliceArg = [Si];
    fn ndim(&self) -> usize { self.len() }
    fn slice(&self) -> &[Ix] { self }
    fn slice_mut(&mut self) -> &mut [Ix] { self }
}

/// Array shape with a next smaller dimension.
///
/// `RemoveAxis` defines a larger-than relation for array shapes:
/// removing one axis from *Self* gives smaller dimension *Smaller*.
pub trait RemoveAxis : Dimension {
    type Smaller: Dimension;
    fn remove_axis(&self, axis: Axis) -> Self::Smaller;
}

macro_rules! impl_shrink(
    ($_a:ident, ) => {}; // implement this case manually below
    ($_a:ident, $_b:ident, ) => {}; // implement this case manually below
    ($from:ident, $($more:ident,)*) => (
impl RemoveAxis for ($from $(,$more)*)
{
    type Smaller = ($($more),*);
    #[allow(unused_parens)]
    #[inline]
    fn remove_axis(&self, axis: Axis) -> ($($more),*) {
        let mut tup = ($(0 as $more),*);
        {
            let mut it = tup.slice_mut().iter_mut();
            for (i, &d) in self.slice().iter().enumerate() {
                if i == axis.axis() {
                    continue;
                }
                for rr in it.by_ref() {
                    *rr = d;
                    break
                }
            }
        }
        tup
    }
}
    )
);

impl RemoveAxis for Ix {
    type Smaller = ();
    #[inline]
    fn remove_axis(&self, _: Axis) { }
}

impl RemoveAxis for (Ix, Ix) {
    type Smaller = Ix;
    #[inline]
    fn remove_axis(&self, axis: Axis) -> Ix {
        let axis = axis.axis();
        debug_assert!(axis < self.ndim());
        if axis == 0 { self.1 } else { self.0 }
    }
}

macro_rules! impl_shrink_recursive(
    ($ix:ident, ) => (impl_shrink!($ix,););
    ($ix1:ident, $($ix:ident,)*) => (
        impl_shrink_recursive!($($ix,)*);
        impl_shrink!($ix1, $($ix,)*);
    )
);

// 12 is the maximum number for having the Eq trait from libstd
impl_shrink_recursive!(Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix,);

impl RemoveAxis for Vec<Ix> {
    type Smaller = Vec<Ix>;
    fn remove_axis(&self, axis: Axis) -> Vec<Ix> {
        let mut res = self.clone();
        res.remove(axis.axis());
        res
    }
}

/// Tuple or fixed size arrays that can be used to index an array.
///
/// ```
/// use ndarray::arr2;
///
/// let mut a = arr2(&[[0, 1], [0, 0]]);
/// a[[1, 1]] = 1;
/// assert_eq!(a[[0, 1]], 1);
/// assert_eq!(a[[1, 1]], 1);
/// ```
///
/// **Note** that `NdIndex` is implemented for all `D where D: Dimension`.
pub unsafe trait NdIndex : Debug {
    type Dim: Dimension;
    #[doc(hidden)]
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize>;
}

unsafe impl<D> NdIndex for D
    where D: Dimension
{
    type Dim = D;
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize> {
        dim.stride_offset_checked(strides, self)
    }
}

unsafe impl NdIndex for [Ix; 0] {
    type Dim = ();
    #[inline]
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize> {
        dim.stride_offset_checked(strides, &())
    }
}

unsafe impl NdIndex for [Ix; 1] {
    type Dim = Ix;
    #[inline]
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize> {
        dim.stride_offset_checked(strides, &self[0])
    }
}

unsafe impl NdIndex for [Ix; 2] {
    type Dim = (Ix, Ix);
    #[inline]
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize> {
        let index = (self[0], self[1]);
        dim.stride_offset_checked(strides, &index)
    }
}

unsafe impl NdIndex for [Ix; 3] {
    type Dim = (Ix, Ix, Ix);
    #[inline]
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize> {
        let index = (self[0], self[1], self[2]);
        dim.stride_offset_checked(strides, &index)
    }
}

unsafe impl NdIndex for [Ix; 4] {
    type Dim = (Ix, Ix, Ix, Ix);
    #[inline]
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize> {
        let index = (self[0], self[1], self[2], self[3]);
        dim.stride_offset_checked(strides, &index)
    }
}

unsafe impl<'a> NdIndex for &'a [Ix] {
    type Dim = Vec<Ix>;
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize> {
        let mut offset = 0;
        for (&d, &i, &s) in zipsl(&dim[..], &self[..]).zip_cons(strides.slice()) {
            if i >= d {
                return None;
            }
            offset += stride_offset(i, s);
        }
        Some(offset)
    }
}

// NOTE: These tests are not compiled & tested
#[cfg(test)]
mod test {
    use super::Dimension;
    use error::{from_kind, ErrorKind};

    #[test]
    fn slice_indexing_uncommon_strides() {
        let v: Vec<_> = (0..12).collect();
        let dim = (2, 3, 2);
        let strides = (1, 2, 6);
        assert!(super::can_index_slice(&v, &dim, &strides).is_ok());

        let strides = (2, 4, 12);
        assert_eq!(super::can_index_slice(&v, &dim, &strides),
                   Err(from_kind(ErrorKind::OutOfBounds)));
    }

    #[test]
    fn overlapping_strides_dim() {
        let dim = (2, 3, 2);
        let strides = (5, 2, 1);
        assert!(super::dim_stride_overlap(&dim, &strides));
        let strides = (6, 2, 1);
        assert!(!super::dim_stride_overlap(&dim, &strides));
        let strides = (6, 0, 1);
        assert!(super::dim_stride_overlap(&dim, &strides));
    }
}

/// An axis index.
///
/// An axis one of an array’s “dimensions”; an *n*-dimensional array has *n* axes.
/// Axis *0* is the array’s outermost axis and *n*-1 is the innermost.
///
/// All array axis arguments use this type to make the code easier to write
/// correctly and easier to understand.
#[derive(Copy, Eq, Ord, Hash, Debug)]
pub struct Axis(pub usize);

impl Axis {
    #[inline(always)]
    pub fn axis(&self) -> usize { self.0 }
}

macro_rules! clone_from_copy {
    ($typename:ident) => {
        impl Clone for $typename {
            #[inline]
            fn clone(&self) -> Self { *self }
        }
    }
}

macro_rules! derive_cmp {
    ($traitname:ident for $typename:ident, $method:ident -> $ret:ty) => {
        impl $traitname for $typename {
            #[inline(always)]
            fn $method(&self, rhs: &Self) -> $ret {
                (self.0).$method(&rhs.0)
            }
        }
    }
}

derive_cmp!{PartialEq for Axis, eq -> bool}
derive_cmp!{PartialOrd for Axis, partial_cmp -> Option<Ordering>}
clone_from_copy!{Axis}

