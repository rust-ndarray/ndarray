// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::{Axis, Dim, Dimension, Ix, Ix0, Ix1};

/// Array shape with a next smaller dimension.
///
/// `RemoveAxis` defines a larger-than relation for array shapes:
/// removing one axis from *Self* gives smaller dimension *Smaller*.
pub trait RemoveAxis: Dimension {
    fn remove_axis(&self, axis: Axis) -> Self::Smaller;
}

impl RemoveAxis for Dim<[Ix; 1]> {
    #[inline]
    fn remove_axis(&self, axis: Axis) -> Ix0 {
        debug_assert!(axis.index() < self.ndim());
        Ix0()
    }
}

impl RemoveAxis for Dim<[Ix; 2]> {
    #[inline]
    fn remove_axis(&self, axis: Axis) -> Ix1 {
        let axis = axis.index();
        debug_assert!(axis < self.ndim());
        if axis == 0 {
            Ix1(get!(self, 1))
        } else {
            Ix1(get!(self, 0))
        }
    }
}

macro_rules! impl_remove_axis_array(
    ($($n:expr),*) => (
    $(
        impl RemoveAxis for Dim<[Ix; $n]>
        {
            #[inline]
            fn remove_axis(&self, axis: Axis) -> Self::Smaller {
                debug_assert!(axis.index() < self.ndim());
                let mut tup = Dim([0; $n - 1]);
                {
                    let mut it = tup.slice_mut().iter_mut();
                    for (i, &d) in self.slice().iter().enumerate() {
                        if i != axis.index() {
                            // FIXME: is this a correct translations from the existing version?
                            *it.next().unwrap() = d;
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
