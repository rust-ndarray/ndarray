// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::imp_prelude::*;
use crate::dimension::IntoDimension;
use crate::dimension::broadcast::upcast;

impl<'a, A, D> ArrayView<'a, A, D>
where
    D: Dimension,
{
    /// Broadcasts an `ArrayView`. See [`ArrayBase::broadcast`].
    ///
    /// This is a specialized version of [`ArrayBase::broadcast`] that transfers
    /// the view's lifetime to the output.
    pub fn broadcast_ref<E>(&self, dim: E) -> Option<ArrayView<'a, A, E::Dim>>
    where
        E: IntoDimension,
    {
        let dim = dim.into_dimension();

        // Note: zero strides are safe precisely because we return an read-only view
        let broadcast_strides = match upcast(&dim, &self.dim, &self.strides) {
            Some(st) => st,
            None => return None,
        };
        unsafe { Some(ArrayView::new(self.ptr, dim, broadcast_strides)) }
    }
}
