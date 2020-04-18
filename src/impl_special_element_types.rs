// Copyright 2020 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem::size_of;
use std::mem::ManuallyDrop;
use std::mem::MaybeUninit;

use crate::imp_prelude::*;
use crate::RawDataSubst;


/// Methods specific to arrays with `MaybeUninit` elements.
///
/// ***See also all methods for [`ArrayBase`]***
///
/// [`ArrayBase`]: struct.ArrayBase.html
impl<A, S, D> ArrayBase<S, D>
where
    S: RawDataSubst<A, Elem=MaybeUninit<A>>,
    D: Dimension,
{
    /// **Promise** that the array's elements are all fully initialized, and convert
    /// the array from element type `MaybeUninit<A>` to `A`.
    ///
    /// For example, it can convert an `Array<MaybeUninit<f64>, D>` to `Array<f64, D>`.
    ///
    /// ## Safety
    ///
    /// Safe to use if all the array's elements have been initialized.
    ///
    /// Note that for owned and shared ownership arrays, the promise must include all of the
    /// array's storage; it is for example possible to slice these in place, but that must
    /// only be done after all elements have been initialized.
    pub unsafe fn assume_init(self) -> ArrayBase<<S as RawDataSubst<A>>::Output, D> {
        // NOTE: Fully initialized includes elements not reachable in current slicing/view.

        let ArrayBase { data, ptr, dim, strides } = self;

        // transmute from storage of MaybeUninit<A> to storage of A
        let data = unlimited_transmute::<S, S::Output>(data);
        let ptr = ptr.cast::<A>();

        ArrayBase {
            data,
            ptr,
            dim,
            strides,
        }
    }
}

/// Transmute from A to B.
///
/// Like transmute, but does not have the compile-time size check which blocks
/// using regular transmute for "S to S::Output".
///
/// **Panics** if the size of A and B are different.
unsafe fn unlimited_transmute<A, B>(data: A) -> B {
    assert_eq!(size_of::<A>(), size_of::<B>());
    let old_data = ManuallyDrop::new(data);
    (&*old_data as *const A as *const B).read()
}
