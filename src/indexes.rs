// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use {ArrayBase, Data};
use super::Dimension;
use dimension::IntoDimension;

/// An iterator over the indexes of an array shape.
///
/// Iterator element type is `D`.
#[derive(Clone)]
pub struct Indices<D> {
    dim: D,
    index: Option<D>,
}

/// Create an iterator over the array shape `shape`.
///
/// *Note:* prefer higher order methods, arithmetic operations and
/// non-indexed iteration before using indices.
pub fn indices<E>(shape: E) -> Indices<E::Dim>
    where E: IntoDimension,
{
    let dim = shape.into_dimension();
    Indices {
        index: dim.first_index(),
        dim: dim,
    }
}

/// Create an iterator over the indices of the passed-in array.
///
/// *Note:* prefer higher order methods, arithmetic operations and
/// non-indexed iteration before using indices.
pub fn indices_of<S, D>(array: &ArrayBase<S, D>) -> Indices<D>
    where S: Data, D: Dimension,
{
    indices(array.dim())
}

impl<D> Iterator for Indices<D>
    where D: Dimension,
{
    type Item = D::Pattern;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = match self.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        self.index = self.dim.next_for(index.clone());
        Some(index.into_pattern())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let l = match self.index {
            None => 0,
            Some(ref ix) => {
                let gone = self.dim
                               .default_strides()
                               .slice()
                               .iter()
                               .zip(ix.slice().iter())
                               .fold(0, |s, (&a, &b)| s + a as usize * b as usize);
                self.dim.size() - gone
            }
        };
        (l, Some(l))
    }
}

impl<D> ExactSizeIterator for Indices<D>
    where D: Dimension
{}
