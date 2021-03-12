use crate::error::*;
use crate::{Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

/// Calculate the common shape for a pair of array shapes, that they can be broadcasted
/// to. Return an error if the shapes are not compatible.
///
/// Uses the [NumPy broadcasting rules]
//  (https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules).
fn co_broadcast<D1, D2, Output>(shape1: &D1, shape2: &D2) -> Result<Output, ShapeError>
    where
        D1: Dimension,
        D2: Dimension,
        Output: Dimension,
{
    let (k, overflow) = shape1.ndim().overflowing_sub(shape2.ndim());
    // Swap the order if d2 is longer.
    if overflow {
        return co_broadcast::<D2, D1, Output>(shape2, shape1);
    }
    // The output should be the same length as shape1.
    let mut out = Output::zeros(shape1.ndim());
    for (out, s) in izip!(out.slice_mut(), shape1.slice()) {
        *out = *s;
    }
    for (out, s2) in izip!(&mut out.slice_mut()[k..], shape2.slice()) {
        if *out != *s2 {
            if *out == 1 {
                *out = *s2
            } else if *s2 != 1 {
                return Err(from_kind(ErrorKind::IncompatibleShape));
            }
        }
    }
    Ok(out)
}

pub trait DimMax<Other: Dimension> {
    /// The resulting dimension type after broadcasting.
    type Output: Dimension;

    /// Determines the shape after broadcasting the shapes together.
    ///
    /// If the shapes are not compatible, returns `Err`.
    fn broadcast_shape(&self, other: &Other) -> Result<Self::Output, ShapeError>;
}

/// Dimensions of the same type remain unchanged when co_broadcast.
/// So you can directly use D as the resulting type.
/// (Instead of <D as DimMax<D>>::BroadcastOutput)
impl<D: Dimension> DimMax<D> for D {
    type Output = D;

    fn broadcast_shape(&self, other: &D) -> Result<Self::Output, ShapeError> {
        co_broadcast::<D, D, Self::Output>(self, other)
    }
}

macro_rules! impl_broadcast_distinct_fixed {
    ($smaller:ty, $larger:ty) => {
        impl DimMax<$larger> for $smaller {
            type Output = $larger;

            fn broadcast_shape(&self, other: &$larger) -> Result<Self::Output, ShapeError> {
                co_broadcast::<Self, $larger, Self::Output>(self, other)
            }
        }

        impl DimMax<$smaller> for $larger {
            type Output = $larger;

            fn broadcast_shape(&self, other: &$smaller) -> Result<Self::Output, ShapeError> {
                co_broadcast::<Self, $smaller, Self::Output>(self, other)
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
