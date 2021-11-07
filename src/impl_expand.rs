// Copyright 2021 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::imp_prelude::*;

use crate::data_traits::RawDataSubst;

use num_complex::Complex;


pub unsafe trait MultiElement {
    type Elem;
    const LEN: usize;
}

unsafe impl<A, const N: usize> MultiElement for [A; N] {
    type Elem = A;
    const LEN: usize = N;
}

unsafe impl<A> MultiElement for Complex<A> {
    type Elem = A;
    const LEN: usize = 2;
}

impl<'a, A, D> ArrayView<'a, A, D>
where
    D: Dimension,
    A: MultiElement,
{
    ///
    /// Note: expanding a zero-element array leads to a new axis of length zero,
    /// i.e. the array becomes empty.
    ///
    /// **Panics** if the product of non-zero axis lengths overflows `isize`.
    pub fn expand(self, new_axis: Axis) -> ArrayView<'a, A::Elem, D::Larger> {
        let mut strides = self.strides.insert_axis(new_axis);
        let mut dim = self.dim.insert_axis(new_axis);
        let len = A::LEN as isize;
        for ax in 0..strides.ndim() {
            if Axis(ax) == new_axis {
                continue;
            }
            if dim[ax] > 1 {
                strides[ax] = ((strides[ax] as isize) * len) as usize;
            }
        }
        dim[new_axis.index()] = A::LEN;
        // TODO nicer assertion
        crate::dimension::size_of_shape_checked(&dim).unwrap();

        // safe because
        // size still fits in isize;
        // new strides are adapted to new element type, inside the same allocation.
        unsafe {
            ArrayBase::from_data_ptr(self.data.data_subst(), self.ptr.cast())
                .with_strides_dim(strides, dim)
        }
    }
}
