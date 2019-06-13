// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use super::Dimension;
use crate::dimension::IntoDimension;
use crate::zip::{Offset, Splittable};
use crate::Axis;
use crate::Layout;
use crate::NdProducer;
use crate::{ArrayBase, Data};

/// An iterator over the indexes of an array shape.
///
/// Iterator element type is `D`.
#[derive(Clone)]
pub struct IndicesIter<D> {
    dim: D,
    index: Option<D>,
}

/// Create an iterable of the array shape `shape`.
///
/// *Note:* prefer higher order methods, arithmetic operations and
/// non-indexed iteration before using indices.
pub fn indices<E>(shape: E) -> Indices<E::Dim>
where
    E: IntoDimension,
{
    let dim = shape.into_dimension();
    Indices {
        start: E::Dim::zeros(dim.ndim()),
        dim,
    }
}

/// Return an iterable of the indices of the passed-in array.
///
/// *Note:* prefer higher order methods, arithmetic operations and
/// non-indexed iteration before using indices.
pub fn indices_of<S, D>(array: &ArrayBase<S, D>) -> Indices<D>
where
    S: Data,
    D: Dimension,
{
    indices(array.dim())
}

impl<D> Iterator for IndicesIter<D>
where
    D: Dimension,
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
                let gone = self
                    .dim
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

impl<D> ExactSizeIterator for IndicesIter<D> where D: Dimension {}

impl<D> IntoIterator for Indices<D>
where
    D: Dimension,
{
    type Item = D::Pattern;
    type IntoIter = IndicesIter<D>;
    fn into_iter(self) -> Self::IntoIter {
        let sz = self.dim.size();
        let index = if sz != 0 { Some(self.start) } else { None };
        IndicesIter {
            index,
            dim: self.dim,
        }
    }
}

/// Indices producer and iterable.
///
/// `Indices` is an `NdProducer` that produces the indices of an array shape.
#[derive(Copy, Clone, Debug)]
pub struct Indices<D>
where
    D: Dimension,
{
    start: D,
    dim: D,
}

#[derive(Copy, Clone, Debug)]
pub struct IndexPtr<D> {
    index: D,
}

impl<D> Offset for IndexPtr<D>
where
    D: Dimension + Copy,
{
    // stride: The axis to increment
    type Stride = usize;

    unsafe fn stride_offset(mut self, stride: Self::Stride, index: usize) -> Self {
        self.index[stride] += index;
        self
    }
    private_impl! {}
}

impl<D: Dimension + Copy> NdProducer for Indices<D> {
    type Item = D::Pattern;
    type Dim = D;
    type Ptr = IndexPtr<D>;
    type Stride = usize;

    private_impl! {}

    #[doc(hidden)]
    #[allow(clippy::clone_on_copy)]
    fn raw_dim(&self) -> Self::Dim {
        self.dim.clone()
    }

    #[doc(hidden)]
    fn equal_dim(&self, dim: &Self::Dim) -> bool {
        self.dim.equal(dim)
    }

    #[doc(hidden)]
    fn as_ptr(&self) -> Self::Ptr {
        IndexPtr { index: self.start }
    }

    #[doc(hidden)]
    fn layout(&self) -> Layout {
        if self.dim.ndim() <= 1 {
            Layout::one_dimensional()
        } else {
            Layout::none()
        }
    }

    #[doc(hidden)]
    unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
        ptr.index.into_pattern()
    }

    #[doc(hidden)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
        let mut index = *i;
        index += &self.start;
        IndexPtr { index }
    }

    #[doc(hidden)]
    fn stride_of(&self, axis: Axis) -> Self::Stride {
        axis.index()
    }

    #[inline(always)]
    fn contiguous_stride(&self) -> Self::Stride {
        0
    }

    #[doc(hidden)]
    fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
        let start_a = self.start;
        let mut start_b = start_a;
        let (a, b) = self.dim.split_at(axis, index);
        start_b[axis.index()] += index;
        (
            Indices {
                start: start_a,
                dim: a,
            },
            Indices {
                start: start_b,
                dim: b,
            },
        )
    }
}

/// An iterator over the indexes of an array shape.
///
/// Iterator element type is `D`.
#[derive(Clone)]
pub struct IndicesIterF<D> {
    dim: D,
    index: D,
    has_remaining: bool,
}

pub fn indices_iter_f<E>(shape: E) -> IndicesIterF<E::Dim>
where
    E: IntoDimension,
{
    let dim = shape.into_dimension();
    let zero = E::Dim::zeros(dim.ndim());
    IndicesIterF {
        has_remaining: dim.size_checked() != Some(0),
        index: zero,
        dim,
    }
}

impl<D> Iterator for IndicesIterF<D>
where
    D: Dimension,
{
    type Item = D::Pattern;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_remaining {
            None
        } else {
            let elt = self.index.clone().into_pattern();
            self.has_remaining = self.dim.next_for_f(&mut self.index);
            Some(elt)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if !self.has_remaining {
            return (0, Some(0));
        }
        let l = match self.index {
            ref ix => {
                let gone = self
                    .dim
                    .fortran_strides()
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

impl<D> ExactSizeIterator for IndicesIterF<D> where D: Dimension {}

#[cfg(test)]
mod tests {
    use super::indices;
    use super::indices_iter_f;

    #[test]
    fn test_indices_iter_c_size_hint() {
        let dim = (3, 4);
        let mut it = indices(dim).into_iter();
        let mut len = dim.0 * dim.1;
        assert_eq!(it.len(), len);
        while let Some(_) = it.next() {
            len -= 1;
            assert_eq!(it.len(), len);
        }
        assert_eq!(len, 0);
    }

    #[test]
    fn test_indices_iter_f_size_hint() {
        let dim = (3, 4);
        let mut it = indices_iter_f(dim);
        let mut len = dim.0 * dim.1;
        assert_eq!(it.len(), len);
        while let Some(_) = it.next() {
            len -= 1;
            assert_eq!(it.len(), len);
        }
        assert_eq!(len, 0);
    }
}
