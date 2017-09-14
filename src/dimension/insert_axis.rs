// Copyright 2014-2017 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use {Ix, IxDynImpl, Dimension, Dim, Axis};

/// Array shape with a next larger dimension.
///
/// `InsertAxis` defines a smaller-than relation for array shapes:
/// inserting one axis into *Self* gives larger dimension *Self::Larger*.
pub trait InsertAxis: Dimension {
    /// Insert a dimension (with length 1) at the specified index.
    fn insert_axis(&self, axis: Axis) -> Self::Larger;
}

macro_rules! impl_insert_axis_array(
    ($($n:expr),*) => (
    $(
        impl InsertAxis for Dim<[Ix; $n]>
        {
            #[inline]
            fn insert_axis(&self, axis: Axis) -> Self::Larger {
                debug_assert!(axis.index() <= self.ndim());
                let mut new = [1; $n + 1];
                new[0..axis.index()].copy_from_slice(&self.slice()[0..axis.index()]);
                new[axis.index()+1..$n+1].copy_from_slice(&self.slice()[axis.index()..$n]);
                Dim(new)
            }
        }
    )*
    );
);

impl_insert_axis_array!(0, 1, 2, 3, 4, 5);

macro_rules! impl_insert_axis_dyn(
    ($($name:ty),*) => (
    $(
        impl InsertAxis for $name {
            #[inline]
            fn insert_axis(&self, axis: Axis) -> Self::Larger {
                debug_assert!(axis.index() <= self.ndim());
                let mut new = Vec::with_capacity(self.ndim() + 1);
                new.extend_from_slice(&self.slice()[0..axis.index()]);
                new.push(1);
                new.extend_from_slice(&self.slice()[axis.index()..self.ndim()]);
                Dim(new)
            }
        }
    )*
    );
);

impl_insert_axis_dyn!(Dim<[Ix; 6]>, Dim<IxDynImpl>);
