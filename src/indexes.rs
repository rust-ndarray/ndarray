// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use super::Dimension;

/// An iterator over the indexes of an array shape.
///
/// Iterator element type is `D`.
#[derive(Clone)]
pub struct Indexes<D> {
    dim: D,
    index: Option<D>,
}

impl<D: Dimension> Indexes<D> {
    /// Create an iterator over the array shape `dim`.
    pub fn new(dim: D) -> Indexes<D> {
        Indexes {
            index: dim.first_index(),
            dim: dim,
        }
    }
}

impl<D> Iterator for Indexes<D>
    where D: Dimension,
{
    type Item = D;
    #[inline]
    fn next(&mut self) -> Option<D> {
        let index = match self.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        self.index = self.dim.next_for(index.clone());
        Some(index)
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

impl<D> ExactSizeIterator for Indexes<D>
    where D: Dimension
{}
