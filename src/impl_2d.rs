// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//! Methods for two-dimensional arrays.
use imp_prelude::*;

/// # Methods For 2-D Arrays
impl<A, S> ArrayBase<S, Ix2>
    where S: Data<Elem=A>,
{
    /// Return an array view of row `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn row(&self, index: Ix) -> ArrayView1<A>
    {
        self.subview(Axis(0), index)
    }

    /// Return a mutable array view of row `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn row_mut(&mut self, index: Ix) -> ArrayViewMut1<A>
        where S: DataMut
    {
        self.index_axis_mut(Axis(0), index)
    }

    /// Return the number of rows (length of `Axis(0)`) in the two-dimensional array.
    pub fn rows(&self) -> usize {
        self.len_of(Axis(0))
    }

    /// Return an array view of column `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn column(&self, index: Ix) -> ArrayView1<A>
    {
        self.subview(Axis(1), index)
    }

    /// Return a mutable array view of column `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn column_mut(&mut self, index: Ix) -> ArrayViewMut1<A>
        where S: DataMut
    {
        self.index_axis_mut(Axis(1), index)
    }

    /// Return the number of columns (length of `Axis(1)`) in the two-dimensional array.
    pub fn cols(&self) -> usize {
        self.len_of(Axis(1))
    }

    /// Return true if the array is square, false otherwise.
    pub fn is_square(&self) -> bool {
        self.rows() == self.cols()
    }
}

