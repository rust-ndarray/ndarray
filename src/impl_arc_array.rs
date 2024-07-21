// Copyright 2019 ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use alloc::sync::Arc;
use crate::imp_prelude::*;

/// Methods specific to `ArcArray`.
///
/// ***See also all methods for [`ArrayBase`]***
impl<A, D> ArcArray<A, D>
where
    D: Dimension,
{
    /// Returns `true` iff the inner `Arc` is not shared.
    pub fn is_unique(&mut self) -> bool {
        Arc::get_mut(&mut self.data.0).is_some()
    }
}
