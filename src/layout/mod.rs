//! Building blocks for describing array layout.
//!
//! This module contains types and traits used to describe how an array is structured in memory.
//! At present, it includes utilities for compactly encoding layout information
//! and abstractions for representing an array’s dimensionality.
//!
//! Over time, this module will also define traits and types for shapes, strides, and complete
//! array layouts, providing a clearer separation between these concerns and enabling more
//! flexible and expressive layout representations.

mod bitset;
pub mod rank;
pub mod ranked;
mod shape;
mod n_repr;
mod dyn_repr;

use core::any::type_name;
use core::error::Error;
use core::fmt::{Debug, Display};
use core::marker::PhantomData;

use crate::layout::ranked::Ranked;

#[allow(deprecated)]
pub use bitset::{Layout, LayoutBitset};
pub use dyn_repr::DShape;
pub use n_repr::NShape;
pub use shape::Shape;

/// The error type for dealing with shapes and strides
#[derive(Debug, Clone, Copy)]
pub enum ShapeStrideError<S>
{
    /// Out of bounds; specifically, using an index that is larger than the dimensionality of the shape or strides `S`.
    OutOfBounds(PhantomData<S>, usize),
    /// The error when trying to construct or mutate a shape or strides with the wrong dimensionality value.
    RankMismatch(PhantomData<S>, usize),
    /// The desired shape would represent an array with more elements than `isize::MAX`
    ShapeOverflow,
}

impl<S: Ranked> Display for ShapeStrideError<S>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
    {
        match self {
            ShapeStrideError::OutOfBounds(_, idx) =>
                write!(f, "Index {idx} is larger than the dimensionality of {}", type_name::<S>()),
            ShapeStrideError::RankMismatch(_, rank) => write!(f, "{} has a rank of {}, which is incompatible with requested rank of {rank}", type_name::<S>(), type_name::<S::NDim>()),
            ShapeStrideError::ShapeOverflow => write!(f, "The desired shape would represent an array with more elements than `usize::MAX`")
        }
    }
}

impl<S: Debug + Ranked> Error for ShapeStrideError<S> {}
