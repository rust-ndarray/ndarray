
//! Producers, iterables and iterators.
//!
//! This module collects all concrete producer, iterable and iterator
//! implementation structs.
//!
//!
//! See also [`NdProducer`](../trait.NdProducer.html).


pub use dimension::{
    Axes,
};
pub use indexes::{
    Indices,
    IndicesIter,
};
pub use iterators::{
    Iter,
    IterMut,
    IndexedIter,
    IndexedIterMut,
    Inners,
    InnersMut,
    InnerIter,
    InnerIterMut,
    AxisIter,
    AxisIterMut,
    AxisChunksIter,
    AxisChunksIterMut,
    ExactChunks,
    ExactChunksIter,
    ExactChunksMut,
    ExactChunksIterMut,
};
