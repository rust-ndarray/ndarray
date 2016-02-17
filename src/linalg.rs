#![allow(non_snake_case, deprecated)]
#![cfg_attr(has_deprecated, deprecated(note="`linalg` is not in good shape."))]

//! ***Deprecated: linalg is not in good shape.***
//!
//! A few linear algebra operations on two-dimensional arrays.

use libnum::{Zero, One};
use std::ops::{Add, Sub, Mul, Div};

/// Trait union for a ring with 1.
pub trait Ring : Clone + Zero + Add<Output=Self> + Sub<Output=Self>
    + One + Mul<Output=Self> { }
impl<A: Clone + Zero + Add<Output=A> + Sub<Output=A> + One + Mul<Output=A>> Ring for A { }

/// Trait union for a field.
pub trait Field : Ring + Div<Output=Self> { }
impl<A: Ring + Div<Output = A>> Field for A {}
