use crate::error::*;
use crate::{Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

/// Calculate the co_broadcast shape of two dimensions. Return error if shapes are
/// not compatible.
fn broadcast_shape<D1, D2, Output>(shape1: &D1, shape2: &D2) -> Result<Output, ShapeError>
where
    D1: Dimension,
    D2: Dimension,
    Output: Dimension,
{
    let (k, overflow) = shape1.ndim().overflowing_sub(shape2.ndim());
    // Swap the order if d2 is longer.
    if overflow {
        return broadcast_shape::<D2, D1, Output>(shape2, shape1);
    }
    // The output should be the same length as shape1.
    let mut out = Output::zeros(shape1.ndim());
    let out_slice = out.slice_mut();
    let s1 = shape1.slice();
    let s2 = shape2.slice();
    // Uses the [NumPy broadcasting rules]
    // (https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules).
    //
    // Zero dimension element is not in the original rules of broadcasting.
    // We currently treat it as the same as 1. Especially, when one side is
    // zero with one side is empty, or both sides are zero, the result will
    // remain zero.
    for i in 0..shape1.ndim() {
        out_slice[i] = s1[i];
    }
    for i in 0..shape2.ndim() {
        if out_slice[i + k] != s2[i] && s2[i] != 0 {
            if out_slice[i + k] <= 1 {
                out_slice[i + k] = s2[i]
            } else if s2[i] != 1 {
                return Err(from_kind(ErrorKind::IncompatibleShape));
            }
        }
    }
    Ok(out)
}

pub trait BroadcastShape<Other: Dimension> {
    /// The resulting dimension type after broadcasting.
    type BroadcastOutput: Dimension;

    /// Determines the shape after broadcasting the dimensions together.
    ///
    /// If the dimensions are not compatible, returns `Err`.
    ///
    /// Uses the [NumPy broadcasting rules]
    /// (https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules).
    fn broadcast_shape(&self, other: &Other) -> Result<Self::BroadcastOutput, ShapeError>;
}

/// Dimensions of the same type remain unchanged when co_broadcast.
/// So you can directly use D as the resulting type.
/// (Instead of <D as BroadcastShape<D>>::BroadcastOutput)
impl<D: Dimension> BroadcastShape<D> for D {
    type BroadcastOutput = D;

    fn broadcast_shape(&self, other: &D) -> Result<Self::BroadcastOutput, ShapeError> {
        broadcast_shape::<D, D, Self::BroadcastOutput>(self, other)
    }
}

macro_rules! impl_broadcast_distinct_fixed {
    ($smaller:ty, $larger:ty) => {
        impl BroadcastShape<$larger> for $smaller {
            type BroadcastOutput = $larger;

            fn broadcast_shape(&self, other: &$larger) -> Result<Self::BroadcastOutput, ShapeError> {
                broadcast_shape::<Self, $larger, Self::BroadcastOutput>(self, other)
            }
        }

        impl BroadcastShape<$smaller> for $larger {
            type BroadcastOutput = $larger;

            fn broadcast_shape(&self, other: &$smaller) -> Result<Self::BroadcastOutput, ShapeError> {
                broadcast_shape::<Self, $smaller, Self::BroadcastOutput>(self, other)
            }
        }
    };
}

impl_broadcast_distinct_fixed!(Ix0, Ix1);
impl_broadcast_distinct_fixed!(Ix0, Ix2);
impl_broadcast_distinct_fixed!(Ix0, Ix3);
impl_broadcast_distinct_fixed!(Ix0, Ix4);
impl_broadcast_distinct_fixed!(Ix0, Ix5);
impl_broadcast_distinct_fixed!(Ix0, Ix6);
impl_broadcast_distinct_fixed!(Ix1, Ix2);
impl_broadcast_distinct_fixed!(Ix1, Ix3);
impl_broadcast_distinct_fixed!(Ix1, Ix4);
impl_broadcast_distinct_fixed!(Ix1, Ix5);
impl_broadcast_distinct_fixed!(Ix1, Ix6);
impl_broadcast_distinct_fixed!(Ix2, Ix3);
impl_broadcast_distinct_fixed!(Ix2, Ix4);
impl_broadcast_distinct_fixed!(Ix2, Ix5);
impl_broadcast_distinct_fixed!(Ix2, Ix6);
impl_broadcast_distinct_fixed!(Ix3, Ix4);
impl_broadcast_distinct_fixed!(Ix3, Ix5);
impl_broadcast_distinct_fixed!(Ix3, Ix6);
impl_broadcast_distinct_fixed!(Ix4, Ix5);
impl_broadcast_distinct_fixed!(Ix4, Ix6);
impl_broadcast_distinct_fixed!(Ix5, Ix6);
impl_broadcast_distinct_fixed!(Ix0, IxDyn);
impl_broadcast_distinct_fixed!(Ix1, IxDyn);
impl_broadcast_distinct_fixed!(Ix2, IxDyn);
impl_broadcast_distinct_fixed!(Ix3, IxDyn);
impl_broadcast_distinct_fixed!(Ix4, IxDyn);
impl_broadcast_distinct_fixed!(Ix5, IxDyn);
impl_broadcast_distinct_fixed!(Ix6, IxDyn);
