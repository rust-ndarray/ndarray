
//! Producers, iterables and iterators.
//!
//! This module collects all concrete producer, iterable and iterator
//! implementation structs.
//!
//!
//! See also [`NdProducer`](../trait.NdProducer.html).


pub use crate::dimension::{
    Axes,
};
pub use crate::indexes::{
    Indices,
    IndicesIter,
};
pub use crate::iterators::{
    Iter,
    IterMut,
    IndexedIter,
    IndexedIterMut,
    Lanes,
    LanesMut,
    LanesIter,
    LanesIterMut,
    AxisIter,
    AxisIterMut,
    AxisChunksIter,
    AxisChunksIterMut,
    ExactChunks,
    ExactChunksIter,
    ExactChunksMut,
    ExactChunksIterMut,
    Windows
};
