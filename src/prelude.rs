// Copyright 2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! ndarray prelude.
//!
//! This module contains the most used types, type aliases, traits and
//! functions that you can import easily as a group.
//!
//! ```
//! extern crate ndarray;
//!
//! use ndarray::prelude::*;
//! # fn main() { }
//! ```

#[doc(no_inline)]
pub use {
    ArrayBase,
    OwnedArray,
    RcArray,
    ArrayView,
    ArrayViewMut,
};
#[doc(no_inline)]
pub use {
    Axis,
    Ix, Ixs,
    Dimension,
};
#[doc(no_inline)]
pub use {
    NdFloat,
    AsArray,
};
#[doc(no_inline)]
pub use {
    arr1, arr2,
    aview0, aview1, aview2,
};

#[doc(no_inline)]
pub use {
    ShapeBuilder,
};
