// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use {Ix, Ixs};
use error::{from_kind, ErrorKind, ShapeError};

pub use self::dim::*;
pub use self::axis::Axis;
pub use self::conversion::IntoDimension;
pub use self::dimension_trait::Dimension;
pub use self::ndindex::NdIndex;
pub use self::remove_axis::RemoveAxis;
pub use self::axes::{axes_of, Axes, AxisDescription};
pub use self::dynindeximpl::IxDynImpl;

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
    for index in last_index.slice_mut().iter_mut() {
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
    for (&d, &i, &s) in izip!(dim.slice(), index.slice(), strides.slice()) {
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
pub fn do_sub<A, D: Dimension>(dims: &mut D, ptr: &mut *mut A, strides: &D,
                               axis: usize, index: Ix) {
    let dim = dims.slice()[axis];
    let stride = strides.slice()[axis];
    ndassert!(index < dim,
              concat!("subview: Index {} must be less than axis length {} ",
                      "for array with shape {:?}"),
             index, dim, *dims);
    dims.slice_mut()[axis] = 1;
    let off = stride_offset(index, stride);
    unsafe {
        *ptr = ptr.offset(off);
    }
}


pub fn merge_axes<D>(dim: &mut D, strides: &mut D, take: Axis, into: Axis) -> bool
    where D: Dimension,
{
    let il = dim.axis(into);
    let is = strides.axis(into) as Ixs;
    let tl = dim.axis(take);
    let ts = strides.axis(take) as Ixs;
    if il as Ixs * is != ts {
        return false;
    }
    // merge them
    dim.set_axis(into, il * tl);
    dim.set_axis(take, 1);
    true
}


// NOTE: These tests are not compiled & tested
#[cfg(test)]
mod test {
    use super::IntoDimension;
    use error::{from_kind, ErrorKind};

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
    }
}

