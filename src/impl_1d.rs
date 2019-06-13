// Copyright 2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Methods for one-dimensional arrays.
use crate::imp_prelude::*;

/// # Methods For 1-D Arrays
impl<A, S> ArrayBase<S, Ix1>
where
    S: RawData<Elem = A>,
{
    /// Return an vector with the elements of the one-dimensional array.
    // TODO: See below re error
    #[allow(clippy::map_clone)]
    pub fn to_vec(&self) -> Vec<A>
    where
        A: Clone,
        S: Data,
    {
        if let Some(slc) = self.as_slice() {
            slc.to_vec()
        } else {
            // clippy suggests this but
            // the trait `iterators::TrustedIterator` is not implemented for `std::iter::Cloned<iterators::Iter<'_, A, dimension::dim::Dim<[usize; 1]>>>`
            // crate::iterators::to_vec(self.iter().cloned())
            crate::iterators::to_vec(self.iter().map(|x| x.clone()))
        }
    }
}
