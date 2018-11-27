// Copyright 2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! ndarray prelude.
//!
//! This module contains the most used types, type aliases, traits, functions,
//! and macros that you can import easily as a group.
//!
//! ```
//! extern crate ndarray;
//!
//! use ndarray::prelude::*;
//! # fn main() { }
//! ```

#[doc(no_inline)]
#[allow(deprecated)]
pub use {
    ArrayBase,
    Array,
    ArcArray,
    RcArray,
    ArrayView,
    ArrayViewMut,
};

#[doc(no_inline)]
pub use {
    Axis,
    Dim,
    Dimension,
};

#[doc(no_inline)]
pub use {Array0, Array1, Array2, Array3, Array4, Array5, Array6, ArrayD};

#[doc(no_inline)]
pub use {ArrayView0, ArrayView1, ArrayView2, ArrayView3, ArrayView4, ArrayView5,
ArrayView6, ArrayViewD};

#[doc(no_inline)]
pub use {ArrayViewMut0, ArrayViewMut1, ArrayViewMut2, ArrayViewMut3,
ArrayViewMut4, ArrayViewMut5, ArrayViewMut6, ArrayViewMutD};

#[doc(no_inline)]
pub use {Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

#[doc(no_inline)]
pub use {
    arr0, arr1, arr2,
    aview0, aview1, aview2,
    aview_mut1,
};

pub use {array, azip, s};

#[doc(no_inline)]
pub use {
    ShapeBuilder,
};

#[doc(no_inline)]
pub use {
    NdFloat,
    AsArray,
};
