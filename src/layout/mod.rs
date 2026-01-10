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
pub mod dimensionality;
pub mod ranked;

#[allow(deprecated)]
pub use bitset::{Layout, LayoutBitset};
