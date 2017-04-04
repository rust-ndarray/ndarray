// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use {Ix, Ix0, Ix1, Dimension, Dim, Axis};
use super::DimPrivate;

/// Array shape with a next smaller dimension.
///
/// `RemoveAxis` defines a larger-than relation for array shapes:
/// removing one axis from *Self* gives smaller dimension *Smaller*.
pub trait RemoveAxis : Dimension {
    type Smaller: Dimension;
    fn remove_axis(&self, axis: Axis) -> Self::Smaller;
}

impl RemoveAxis for Dim<[Ix; 1]> {
    type Smaller = Ix0;
    #[inline]
    fn remove_axis(&self, _: Axis) -> Ix0 { Ix0() }
}

impl RemoveAxis for Dim<[Ix; 2]> {
    type Smaller = Ix1;
    #[inline]
    fn remove_axis(&self, axis: Axis) -> Ix1 {
        let axis = axis.index();
        debug_assert!(axis < self.ndim());
        if axis == 0 { Ix1(get!(self, 1)) } else { Ix1(get!(self, 0)) }
    }
}

macro_rules! impl_remove_axis_array(
    ($($n:expr),*) => (
    $(
        impl RemoveAxis for Dim<[Ix; $n]>
        {
            type Smaller = Dim<[Ix; $n - 1]>;
            #[inline]
            fn remove_axis(&self, axis: Axis) -> Self::Smaller {
                let mut tup = Dim([0; $n - 1]);
                {
                    let mut it = tup.slice_mut().iter_mut();
                    for (i, &d) in self.slice().iter().enumerate() {
                        if i == axis.index() {
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
    )*
    );
);

impl_remove_axis_array!(3, 4, 5, 6);



