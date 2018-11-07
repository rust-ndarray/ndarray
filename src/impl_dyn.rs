// Copyright 2018 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Methods for dynamic-dimensional arrays.
use imp_prelude::*;

/// # Methods for Dynamic-Dimensional Arrays
impl<A, S> ArrayBase<S, IxDyn>
where
    S: Data<Elem = A>,
{
    /// Insert new array axis of length 1 at `axis`, modifying the shape and
    /// strides in-place.
    ///
    /// **Panics** if the axis is out of bounds.
    ///
    /// ```
    /// use ndarray::{Axis, arr2, arr3};
    ///
    /// let mut a = arr2(&[[1, 2, 3], [4, 5, 6]]).into_dyn();
    /// assert_eq!(a.shape(), &[2, 3]);
    ///
    /// a.insert_axis_inplace(Axis(1));
    /// assert_eq!(a, arr3(&[[[1, 2, 3]], [[4, 5, 6]]]).into_dyn());
    /// assert_eq!(a.shape(), &[2, 1, 3]);
    /// ```
    pub fn insert_axis_inplace(&mut self, axis: Axis) {
        assert!(axis.index() <= self.ndim());
        self.dim = self.dim.insert_axis(axis);
        self.strides = self.strides.insert_axis(axis);
    }

    /// Remove array axis `axis`, modifying the shape and strides in-place.
    ///
    /// **Panics** if the axis is out of bounds or its length is zero.
    ///
    /// ```
    /// use ndarray::{Axis, arr1, arr2};
    ///
    /// let mut a = arr2(&[[1, 2, 3], [4, 5, 6]]).into_dyn();
    /// assert_eq!(a.shape(), &[2, 3]);
    ///
    /// a.remove_axis_inplace(Axis(1));
    /// assert_eq!(a, arr1(&[1, 4]).into_dyn());
    /// assert_eq!(a.shape(), &[2]);
    /// ```
    pub fn remove_axis_inplace(&mut self, axis: Axis) {
        let len = self.len_of(axis);
        assert_ne!(len, 0, "Length of removed axis must be nonzero.");
        self.dim = self.dim.remove_axis(axis);
        self.strides = self.strides.remove_axis(axis);
    }
}
