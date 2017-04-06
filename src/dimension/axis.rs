// Copyright 2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::Ordering;

/// An axis index.
///
/// An axis one of an array’s “dimensions”; an *n*-dimensional array has *n* axes.
/// Axis *0* is the array’s outermost axis and *n*-1 is the innermost.
///
/// All array axis arguments use this type to make the code easier to write
/// correctly and easier to understand.
#[derive(Eq, Ord, Hash, Debug)]
pub struct Axis(pub usize);

impl Axis {
    /// Return the index of the axis.
    #[inline(always)]
    pub fn index(&self) -> usize { self.0 }
}

copy_and_clone!{Axis}

macro_rules! derive_cmp {
    ($traitname:ident for $typename:ident, $method:ident -> $ret:ty) => {
        impl $traitname for $typename {
            #[inline(always)]
            fn $method(&self, rhs: &Self) -> $ret {
                (self.0).$method(&rhs.0)
            }
        }
    }
}

derive_cmp!{PartialEq for Axis, eq -> bool}
derive_cmp!{PartialOrd for Axis, partial_cmp -> Option<Ordering>}

