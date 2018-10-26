// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use {Ix, Ixs};
use error::{from_kind, ErrorKind, ShapeError};
use itertools::izip;

pub use self::dim::*;
pub use self::axis::Axis;
pub use self::conversion::IntoDimension;
pub use self::dimension_trait::Dimension;
pub use self::ndindex::NdIndex;
pub use self::remove_axis::RemoveAxis;
pub use self::axes::{axes_of, Axes, AxisDescription};
pub use self::dynindeximpl::IxDynImpl;

use std::isize;
use std::mem;

#[macro_use] mod macros;
mod axis;
mod conversion;
pub mod dim;
mod dimension_trait;
mod dynindeximpl;
mod ndindex;
mod remove_axis;
mod axes;

/// Calculate offset from `Ix` stride converting sign properly
#[inline(always)]
pub fn stride_offset(n: Ix, stride: Ix) -> isize {
    (n as isize) * ((stride as Ixs) as isize)
}

/// Check whether the given `dim` and `stride` lead to overlapping indices
///
/// There is overlap if, when iterating through the dimensions in order of
/// increasing stride, the current stride is less than or equal to the maximum
/// possible offset along the preceding axes. (Axes of length ≤1 are ignored.)
///
/// The current implementation assumes that strides of axes with length > 1 are
/// nonnegative. Additionally, it does not check for overflow.
pub fn dim_stride_overlap<D: Dimension>(dim: &D, strides: &D) -> bool {
    let order = strides._fastest_varying_stride_order();
    let mut sum_prev_offsets = 0;
    for &index in order.slice() {
        let d = dim[index];
        let s = strides[index] as isize;
        match d {
            0 => return false,
            1 => {}
            _ => {
                if s <= sum_prev_offsets {
                    return true;
                }
                sum_prev_offsets += (d - 1) as isize * s;
            }
        }
    }
    false
}

/// Returns the `size` of the `dim`, checking that the product of non-zero axis
/// lengths does not exceed `isize::MAX`.
///
/// If `size_of_checked_shape(dim)` returns `Ok(size)`, the data buffer is a
/// slice or `Vec` of length `size`, and `strides` are created with
/// `self.default_strides()` or `self.fortran_strides()`, then the invariants
/// are met to construct an array from the data buffer, `dim`, and `strides`.
/// (The data buffer being a slice or `Vec` guarantees that it contains no more
/// than `isize::MAX` bytes.)
pub fn size_of_shape_checked<D: Dimension>(dim: &D) -> Result<usize, ShapeError> {
    let size_nonzero = dim
        .slice()
        .iter()
        .filter(|&&d| d != 0)
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| from_kind(ErrorKind::Overflow))?;
    if size_nonzero > ::std::isize::MAX as usize {
        Err(from_kind(ErrorKind::Overflow))
    } else {
        Ok(dim.size())
    }
}

/// Checks whether the given data and dimension meet the invariants of the
/// `ArrayBase` type, assuming the strides are created using
/// `dim.default_strides()` or `dim.fortran_strides()`.
///
/// To meet the invariants,
///
/// 1. The product of non-zero axis lengths must not exceed `isize::MAX`.
///
/// 2. The result of `dim.size()` (assuming no overflow) must be less than or
///    equal to the length of the slice.
///
///    (Since `dim.default_strides()` and `dim.fortran_strides()` always return
///    contiguous strides for non-empty arrays, this ensures that for non-empty
///    arrays the difference between the least address and greatest address
///    accessible by moving along all axes is < the length of the slice. Since
///    `dim.default_strides()` and `dim.fortran_strides()` always return all
///    zero strides for empty arrays, this ensures that for empty arrays the
///    difference between the least address and greatest address accessible by
///    moving along all axes is ≤ the length of the slice.)
///
/// Note that since slices cannot contain more than `isize::MAX` bytes,
/// conditions 1 and 2 are sufficient to guarantee that the offset in units of
/// `A` and in units of bytes between the least address and greatest address
/// accessible by moving along all axes does not exceed `isize::MAX`.
pub fn can_index_slice_not_custom<A, D: Dimension>(data: &[A], dim: &D) -> Result<(), ShapeError> {
    // Condition 1.
    let len = size_of_shape_checked(dim)?;
    // Condition 2.
    if len > data.len() {
        return Err(from_kind(ErrorKind::OutOfBounds));
    }
    Ok(())
}

/// Returns the absolute difference in units of `A` between least and greatest
/// address accessible by moving along all axes.
///
/// Returns `Ok` only if
///
/// 1. The ndim of `dim` and `strides` is the same.
///
/// 2. The absolute difference in units of `A` and in units of bytes between
///    the least address and greatest address accessible by moving along all axes
///    does not exceed `isize::MAX`.
///
/// 3. The product of non-zero axis lengths does not exceed `isize::MAX`. (This
///    also implies that the length of any individual axis does not exceed
///    `isize::MAX`.)
pub fn max_abs_offset_check_overflow<A, D>(dim: &D, strides: &D) -> Result<usize, ShapeError>
where
    D: Dimension,
{
    // Condition 1.
    if dim.ndim() != strides.ndim() {
        return Err(from_kind(ErrorKind::IncompatibleLayout));
    }

    // Condition 3.
    let _ = size_of_shape_checked(dim)?;

    // Determine absolute difference in units of `A` between least and greatest
    // address accessible by moving along all axes.
    let max_offset: usize = izip!(dim.slice(), strides.slice())
        .try_fold(0usize, |acc, (&d, &s)| {
            let s = s as isize;
            // Calculate maximum possible absolute movement along this axis.
            let off = d.saturating_sub(1).checked_mul(s.abs() as usize)?;
            acc.checked_add(off)
        }).ok_or_else(|| from_kind(ErrorKind::Overflow))?;
    // Condition 2a.
    if max_offset > isize::MAX as usize {
        return Err(from_kind(ErrorKind::Overflow));
    }

    // Determine absolute difference in units of bytes between least and
    // greatest address accessible by moving along all axes
    let max_offset_bytes = max_offset
        .checked_mul(mem::size_of::<A>())
        .ok_or_else(|| from_kind(ErrorKind::Overflow))?;
    // Condition 2b.
    if max_offset_bytes > isize::MAX as usize {
        return Err(from_kind(ErrorKind::Overflow));
    }

    Ok(max_offset)
}

/// Checks whether the given data, dimension, and strides meet the invariants
/// of the `ArrayBase` type (except for checking ownership of the data).
///
/// To meet the invariants,
///
/// 1. The ndim of `dim` and `strides` must be the same.
///
/// 2. The product of non-zero axis lengths must not exceed `isize::MAX`.
///
/// 3. For axes with length > 1, the stride must be nonnegative. This is
///    necessary to make sure the pointer cannot move backwards outside the
///    slice. For axes with length ≤ 1, the stride can be anything.
///
/// 4. If the array will be empty (any axes are zero-length), the difference
///    between the least address and greatest address accessible by moving
///    along all axes must be ≤ `data.len()`. (It's fine in this case to move
///    one byte past the end of the slice since the pointers will be offset but
///    never dereferenced.)
///
///    If the array will not be empty, the difference between the least address
///    and greatest address accessible by moving along all axes must be <
///    `data.len()`. This and #3 ensure that all dereferenceable pointers point
///    to elements within the slice.
///
/// 5. The strides must not allow any element to be referenced by two different
///    indices.
///
/// Note that since slices cannot contain more than `isize::MAX` bytes,
/// condition 4 is sufficient to guarantee that the absolute difference in
/// units of `A` and in units of bytes between the least address and greatest
/// address accessible by moving along all axes does not exceed `isize::MAX`.
pub fn can_index_slice<A, D: Dimension>(data: &[A], dim: &D, strides: &D)
    -> Result<(), ShapeError>
{
    // Check conditions 1 and 2 and calculate `max_offset`.
    let max_offset = max_abs_offset_check_overflow::<A, _>(dim, strides)?;

    // Check condition 4.
    let is_empty = dim.slice().iter().any(|&d| d == 0);
    if is_empty && max_offset > data.len() {
        return Err(from_kind(ErrorKind::OutOfBounds));
    }
    if !is_empty && max_offset >= data.len() {
        return Err(from_kind(ErrorKind::OutOfBounds));
    }

    // Check condition 3.
    for (&d, &s) in izip!(dim.slice(), strides.slice()) {
        let s = s as isize;
        if d > 1 && s < 0 {
            return Err(from_kind(ErrorKind::Unsupported));
        }
    }

    // Check condition 5.
    if !is_empty && dim_stride_overlap(dim, strides) {
        return Err(from_kind(ErrorKind::Unsupported));
    }

    Ok(())
}

/// Stride offset checked general version (slices)
#[inline]
pub fn stride_offset_checked(dim: &[Ix], strides: &[Ix], index: &[Ix]) -> Option<isize> {
    if index.len() != dim.len() {
        return None;
    }
    let mut offset = 0;
    for (&d, &i, &s) in izip!(dim, index, strides) {
        if i >= d {
            return None;
        }
        offset += stride_offset(i, s);
    }
    Some(offset)
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
        self[axis.index()]
    }

    #[inline]
    fn set_axis(&mut self, axis: Axis, value: Ix) {
        self[axis.index()] = value;
    }
}

impl<'a> DimensionExt for [Ix]
{
    #[inline]
    fn axis(&self, axis: Axis) -> Ix {
        self[axis.index()]
    }

    #[inline]
    fn set_axis(&mut self, axis: Axis, value: Ix) {
        self[axis.index()] = value;
    }
}

/// Collapse axis `axis` and shift so that only subarray `index` is
/// available.
///
/// **Panics** if `index` is larger than the size of the axis
// FIXME: Move to Dimension trait
pub fn do_collapse_axis<A, D: Dimension>(
    dims: &mut D,
    ptr: &mut *mut A,
    strides: &D,
    axis: usize,
    index: usize,
) {
    let dim = dims.slice()[axis];
    let stride = strides.slice()[axis];
    ndassert!(index < dim,
              "collapse_axis: Index {} must be less than axis length {} for \
               array with shape {:?}",
             index, dim, *dims);
    dims.slice_mut()[axis] = 1;
    let off = stride_offset(index, stride);
    unsafe {
        *ptr = ptr.offset(off);
    }
}

/// Compute the equivalent unsigned index given the axis length and signed index.
#[inline]
pub fn abs_index(len: Ix, index: Ixs) -> Ix {
    if index < 0 {
        len - (-index as Ix)
    } else {
        index as Ix
    }
}

/// Modify dimension, stride and return data pointer offset
///
/// **Panics** if stride is 0 or if any index is out of bounds.
pub fn do_slice(
    dim: &mut Ix,
    stride: &mut Ix,
    start: Ixs,
    end: Option<Ixs>,
    step: Ixs,
) -> isize {
    let mut offset = 0;

    let axis_len = *dim;
    let start = abs_index(axis_len, start);
    let mut end = abs_index(axis_len, end.unwrap_or(axis_len as Ixs));
    if end < start {
        end = start;
    }

    ndassert!(
        start <= axis_len,
        "Slice begin {} is past end of axis of length {}",
        start,
        axis_len,
    );
    ndassert!(
        end <= axis_len,
        "Slice end {} is past end of axis of length {}",
        end,
        axis_len,
    );

    let m = end - start;
    // stride
    let s = (*stride) as Ixs;

    // Data pointer offset
    offset += stride_offset(start, *stride);
    // Adjust for strides
    ndassert!(step != 0, "Slice stride must not be zero");
    // How to implement negative strides:
    //
    // Increase start pointer by
    // old stride * (old dim - 1)
    // to put the pointer completely in the other end
    if step < 0 {
        offset += stride_offset(m - 1, *stride);
    }

    let s_prim = s * step;

    let d = m / step.abs() as Ix;
    let r = m % step.abs() as Ix;
    let m_prim = d + if r > 0 { 1 } else { 0 };

    // Update dimension and stride coordinate
    *dim = m_prim;
    *stride = s_prim as Ix;

    offset
}

pub fn merge_axes<D>(dim: &mut D, strides: &mut D, take: Axis, into: Axis) -> bool
    where D: Dimension,
{
    let into_len = dim.axis(into);
    let into_stride = strides.axis(into) as isize;
    let take_len = dim.axis(take);
    let take_stride = strides.axis(take) as isize;
    let merged_len = into_len * take_len;
    if take_len <= 1 {
        dim.set_axis(into, merged_len);
        dim.set_axis(take, if merged_len == 0 { 0 } else { 1 });
        true
    } else if into_len <= 1 {
        strides.set_axis(into, take_stride as usize);
        dim.set_axis(into, merged_len);
        dim.set_axis(take, if merged_len == 0 { 0 } else { 1 });
        true
    } else if take_stride == into_len as isize * into_stride {
        dim.set_axis(into, merged_len);
        dim.set_axis(take, 1);
        true
    } else {
        false
    }
}


// NOTE: These tests are not compiled & tested
#[cfg(test)]
mod test {
    use super::{
        can_index_slice, can_index_slice_not_custom, max_abs_offset_check_overflow, IntoDimension
    };
    use error::{from_kind, ErrorKind};
    use {Dimension, Ix0, Ix1, Ix2, Ix3, IxDyn};

    #[test]
    fn slice_indexing_uncommon_strides() {
        let v: Vec<_> = (0..12).collect();
        let dim = (2, 3, 2).into_dimension();
        let strides = (1, 2, 6).into_dimension();
        assert!(super::can_index_slice(&v, &dim, &strides).is_ok());

        let strides = (2, 4, 12).into_dimension();
        assert_eq!(super::can_index_slice(&v, &dim, &strides),
                   Err(from_kind(ErrorKind::OutOfBounds)));
    }

    #[test]
    fn overlapping_strides_dim() {
        let dim = (2, 3, 2).into_dimension();
        let strides = (5, 2, 1).into_dimension();
        assert!(super::dim_stride_overlap(&dim, &strides));
        let strides = (6, 2, 1).into_dimension();
        assert!(!super::dim_stride_overlap(&dim, &strides));
        let strides = (6, 0, 1).into_dimension();
        assert!(super::dim_stride_overlap(&dim, &strides));
        let dim = (2, 2).into_dimension();
        let strides = (3, 2).into_dimension();
        assert!(!super::dim_stride_overlap(&dim, &strides));
    }

    #[test]
    fn max_abs_offset_check_overflow_examples() {
        let dim = (1, ::std::isize::MAX as usize, 1).into_dimension();
        let strides = (1, 1, 1).into_dimension();
        max_abs_offset_check_overflow::<u8, _>(&dim, &strides).unwrap();
        let dim = (1, ::std::isize::MAX as usize, 2).into_dimension();
        let strides = (1, 1, 1).into_dimension();
        max_abs_offset_check_overflow::<u8, _>(&dim, &strides).unwrap_err();
        let dim = (0, 2, 2).into_dimension();
        let strides = (1, ::std::isize::MAX as usize, 1).into_dimension();
        max_abs_offset_check_overflow::<u8, _>(&dim, &strides).unwrap_err();
        let dim = (0, 2, 2).into_dimension();
        let strides = (1, ::std::isize::MAX as usize / 4, 1).into_dimension();
        max_abs_offset_check_overflow::<i32, _>(&dim, &strides).unwrap_err();
    }

    #[test]
    fn can_index_slice_ix0() {
        can_index_slice::<i32, _>(&[1], &Ix0(), &Ix0()).unwrap();
        can_index_slice::<i32, _>(&[], &Ix0(), &Ix0()).unwrap_err();
    }

    #[test]
    fn can_index_slice_ix1() {
        can_index_slice::<i32, _>(&[], &Ix1(0), &Ix1(0)).unwrap();
        can_index_slice::<i32, _>(&[], &Ix1(0), &Ix1(1)).unwrap();
        can_index_slice::<i32, _>(&[], &Ix1(1), &Ix1(0)).unwrap_err();
        can_index_slice::<i32, _>(&[], &Ix1(1), &Ix1(1)).unwrap_err();
        can_index_slice::<i32, _>(&[1], &Ix1(1), &Ix1(0)).unwrap();
        can_index_slice::<i32, _>(&[1], &Ix1(1), &Ix1(2)).unwrap();
        can_index_slice::<i32, _>(&[1], &Ix1(1), &Ix1(-1isize as usize)).unwrap();
        can_index_slice::<i32, _>(&[1], &Ix1(2), &Ix1(1)).unwrap_err();
        can_index_slice::<i32, _>(&[1, 2], &Ix1(2), &Ix1(0)).unwrap_err();
        can_index_slice::<i32, _>(&[1, 2], &Ix1(2), &Ix1(1)).unwrap();
        can_index_slice::<i32, _>(&[1, 2], &Ix1(2), &Ix1(-1isize as usize)).unwrap_err();
    }

    #[test]
    fn can_index_slice_ix2() {
        can_index_slice::<i32, _>(&[], &Ix2(0, 0), &Ix2(0, 0)).unwrap();
        can_index_slice::<i32, _>(&[], &Ix2(0, 0), &Ix2(2, 1)).unwrap();
        can_index_slice::<i32, _>(&[], &Ix2(0, 1), &Ix2(0, 0)).unwrap();
        can_index_slice::<i32, _>(&[], &Ix2(0, 1), &Ix2(2, 1)).unwrap();
        can_index_slice::<i32, _>(&[], &Ix2(0, 2), &Ix2(0, 0)).unwrap();
        can_index_slice::<i32, _>(&[], &Ix2(0, 2), &Ix2(2, 1)).unwrap_err();
        can_index_slice::<i32, _>(&[1], &Ix2(1, 2), &Ix2(5, 1)).unwrap_err();
        can_index_slice::<i32, _>(&[1, 2], &Ix2(1, 2), &Ix2(5, 1)).unwrap();
        can_index_slice::<i32, _>(&[1, 2], &Ix2(1, 2), &Ix2(5, 2)).unwrap_err();
        can_index_slice::<i32, _>(&[1, 2, 3, 4, 5], &Ix2(2, 2), &Ix2(3, 1)).unwrap();
        can_index_slice::<i32, _>(&[1, 2, 3, 4], &Ix2(2, 2), &Ix2(3, 1)).unwrap_err();
    }

    #[test]
    fn can_index_slice_ix3() {
        can_index_slice::<i32, _>(&[], &Ix3(0, 0, 1), &Ix3(2, 1, 3)).unwrap();
        can_index_slice::<i32, _>(&[], &Ix3(1, 1, 1), &Ix3(2, 1, 3)).unwrap_err();
        can_index_slice::<i32, _>(&[1], &Ix3(1, 1, 1), &Ix3(2, 1, 3)).unwrap();
        can_index_slice::<i32, _>(&[1; 11], &Ix3(2, 2, 3), &Ix3(6, 3, 1)).unwrap_err();
        can_index_slice::<i32, _>(&[1; 12], &Ix3(2, 2, 3), &Ix3(6, 3, 1)).unwrap();
    }

    #[test]
    fn can_index_slice_zero_size_elem() {
        can_index_slice::<(), _>(&[], &Ix1(0), &Ix1(1)).unwrap();
        can_index_slice::<(), _>(&[()], &Ix1(1), &Ix1(1)).unwrap();
        can_index_slice::<(), _>(&[(), ()], &Ix1(2), &Ix1(1)).unwrap();

        // These might seem okay because the element type is zero-sized, but
        // there could be a zero-sized type such that the number of instances
        // in existence are carefully controlled.
        can_index_slice::<(), _>(&[], &Ix1(1), &Ix1(1)).unwrap_err();
        can_index_slice::<(), _>(&[()], &Ix1(2), &Ix1(1)).unwrap_err();

        can_index_slice::<(), _>(&[(), ()], &Ix2(2, 1), &Ix2(1, 0)).unwrap();
        can_index_slice::<(), _>(&[], &Ix2(0, 2), &Ix2(0, 0)).unwrap();

        // This case would be probably be sound, but that's not entirely clear
        // and it's not worth the special case code.
        can_index_slice::<(), _>(&[], &Ix2(0, 2), &Ix2(2, 1)).unwrap_err();
    }

    quickcheck! {
        fn can_index_slice_not_custom_same_as_can_index_slice(data: Vec<u8>, dim: Vec<usize>) -> bool {
            let dim = IxDyn(&dim);
            let result = can_index_slice_not_custom(&data, &dim);
            if dim.size_checked().is_none() {
                // Avoid overflow `dim.default_strides()` or `dim.fortran_strides()`.
                result.is_err()
            } else {
                result == can_index_slice(&data, &dim, &dim.default_strides()) &&
                    result == can_index_slice(&data, &dim, &dim.fortran_strides())
            }
        }
    }
}
