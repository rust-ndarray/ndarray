use libnum::{Zero, One, Float};
use std::ops::{Add, Sub, Mul, Div};
use std::any::Any;

/// Trait union for scalars (array elements) that support linear algebra operations.
///
/// `Any` for type-based specialization, `Copy` so that they don't need move
/// semantics or destructors, and the rest are numerical traits.
pub trait LinalgScalar :
    Any +
    Copy +
    Zero + One +
    Add<Output=Self> +
    Sub<Output=Self> +
    Mul<Output=Self> +
    Div<Output=Self>
{ }

impl<T> LinalgScalar for T
    where T:
    Any +
    Copy +
    Zero + One +
    Add<Output=T> +
    Sub<Output=T> +
    Mul<Output=T> +
    Div<Output=T>
{ }
