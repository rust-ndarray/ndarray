// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#[macro_use] mod macros;
mod chunks;
mod windows;
mod lanes;
pub mod iter;

use std::marker::PhantomData;
use std::ptr;

use Ix1;

use super::{Dimension, Ix, Ixs};
use super::{
    ArrayBase,
    Data,
    ArrayView,
    ArrayViewMut,
    RemoveAxis,
    Axis,
    NdProducer,
};

pub use self::windows::{
    Windows,
    windows
};
pub use self::chunks::{
    ExactChunks,
    ExactChunksIter,
    exact_chunks_of,
    ExactChunksMut,
    ExactChunksIterMut,
    exact_chunks_mut_of,
};
pub use self::lanes::{
    new_lanes,
    new_lanes_mut,
    Lanes,
    LanesMut,
};

use std::slice::{self, Iter as SliceIter, IterMut as SliceIterMut};

/// Base for array iterators
///
/// Iterator element type is `&'a A`.
pub struct Baseiter<'a, A: 'a, D> {
    // Can have pub fields because it is not itself pub.
    pub ptr: *mut A,
    pub dim: D,
    pub strides: D,
    pub index: Option<D>,
    pub life: PhantomData<&'a A>,
}


impl<'a, A, D: Dimension> Baseiter<'a, A, D> {
    /// Creating a Baseiter is unsafe, because it can
    /// have any lifetime, be immut or mut, and the
    /// boundary and stride parameters need to be correct to
    /// avoid memory unsafety.
    ///
    /// It must be placed in the correct mother iterator to be safe.
    ///
    /// NOTE: Mind the lifetime, it's arbitrary
    #[inline]
    pub unsafe fn new(ptr: *mut A, len: D, stride: D) -> Baseiter<'a, A, D> {
        Baseiter {
            ptr: ptr,
            index: len.first_index(),
            dim: len,
            strides: stride,
            life: PhantomData,
        }
    }
}

impl<'a, A, D: Dimension> Baseiter<'a, A, D> {
    #[inline]
    pub fn next(&mut self) -> Option<*mut A> {
        let index = match self.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        let offset = D::stride_offset(&index, &self.strides);
        self.index = self.dim.next_for(index);
        unsafe { Some(self.ptr.offset(offset)) }
    }

    #[inline]
    fn next_ref(&mut self) -> Option<&'a A> {
        unsafe { self.next().map(|p| &*p) }
    }

    #[inline]
    fn next_ref_mut(&mut self) -> Option<&'a mut A> {
        unsafe { self.next().map(|p| &mut *p) }
    }

    fn len(&self) -> usize {
        match self.index {
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
        }
    }

    fn fold<Acc, G>(mut self, init: Acc, mut g: G) -> Acc
        where G: FnMut(Acc, *mut A) -> Acc,
    {
        let ndim = self.dim.ndim();
        debug_assert_ne!(ndim, 0);
        let mut accum = init;
        loop {
            if let Some(mut index) = self.index.clone() {
                let stride = self.strides.last_elem() as isize;
                let elem_index = index.last_elem();
                let len = self.dim.last_elem();
                let offset = D::stride_offset(&index, &self.strides);
                unsafe {
                    let row_ptr = self.ptr.offset(offset);
                    for i in 0..(len - elem_index) {
                        accum = g(accum, row_ptr.offset(i as isize * stride));
                    }
                }
                index.set_last_elem(len - 1);
                self.index = self.dim.next_for(index);
            } else {
                break;
            };
        }
        accum
    }
}

impl<'a, A> Baseiter<'a, A, Ix1> {
    #[inline]
    fn next_back(&mut self) -> Option<*mut A> {
        let index = match self.index {
            None => return None,
            Some(ix) => ix,
        };
        self.dim[0] -= 1;
        let offset = <_>::stride_offset(&self.dim, &self.strides);
        if index == self.dim {
            self.index = None;
        }

        unsafe { Some(self.ptr.offset(offset)) }
    }

    #[inline]
    fn next_back_ref(&mut self) -> Option<&'a A> {
        unsafe { self.next_back().map(|p| &*p) }
    }

    #[inline]
    fn next_back_ref_mut(&mut self) -> Option<&'a mut A> {
        unsafe { self.next_back().map(|p| &mut *p) }
    }
}

clone_bounds!(
    ['a, A, D: Clone]
    Baseiter['a, A, D] {
        @copy {
            ptr,
            life,
        }
        dim,
        strides,
        index,
    }
);

clone_bounds!(
    ['a, A, D: Clone]
    ElementsBase['a, A, D] {
        @copy {
        }
        inner,
    }
);

impl<'a, A, D: Dimension> Iterator for ElementsBase<'a, A, D> {
    type Item = &'a A;
    #[inline]
    fn next(&mut self) -> Option<&'a A> {
        self.inner.next_ref()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.len();
        (len, Some(len))
    }

    fn fold<Acc, G>(self, init: Acc, mut g: G) -> Acc
        where G: FnMut(Acc, Self::Item) -> Acc,
    {
        unsafe {
            self.inner.fold(init, move |acc, ptr| g(acc, &*ptr))
        }
    }
}

impl<'a, A> DoubleEndedIterator for ElementsBase<'a, A, Ix1> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a A> {
        self.inner.next_back_ref()
    }
}

impl<'a, A, D> ExactSizeIterator for ElementsBase<'a, A, D>
    where D: Dimension
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

macro_rules! either {
    ($value:expr, $inner:pat => $result:expr) => (
        match $value {
            ElementsRepr::Slice($inner) => $result,
            ElementsRepr::Counted($inner) => $result,
        }
    )
}

macro_rules! either_mut {
    ($value:expr, $inner:ident => $result:expr) => (
        match $value {
            ElementsRepr::Slice(ref mut $inner) => $result,
            ElementsRepr::Counted(ref mut $inner) => $result,
        }
    )
}

clone_bounds!(
    ['a, A, D: Clone]
    Iter['a, A, D] {
        @copy {
        }
        inner,
    }
);

impl<'a, A, D> Iter<'a, A, D>
    where D: Dimension
{
    pub(crate) fn new(self_: ArrayView<'a, A, D>) -> Self {
        Iter {
            inner: if let Some(slc) = self_.into_slice() {
                ElementsRepr::Slice(slc.iter())
            } else {
                ElementsRepr::Counted(self_.into_elements_base())
            },
        }
    }
}



impl<'a, A, D> IterMut<'a, A, D>
    where D: Dimension
{
    pub(crate) fn new(self_: ArrayViewMut<'a, A, D>) -> Self {
        IterMut {
            inner:
            match self_.into_slice_() {
                Ok(x) => ElementsRepr::Slice(x.into_iter()),
                Err(self_) => ElementsRepr::Counted(self_.into_elements_base()),
            }
        }
    }
}

#[derive(Clone)]
pub enum ElementsRepr<S, C> {
    Slice(S),
    Counted(C),
}

/// An iterator over the elements of an array.
///
/// Iterator element type is `&'a A`.
///
/// See [`.iter()`](../struct.ArrayBase.html#method.iter) for more information.
pub struct Iter<'a, A: 'a, D> {
    inner: ElementsRepr<SliceIter<'a, A>, ElementsBase<'a, A, D>>,
}

/// Counted read only iterator
pub struct ElementsBase<'a, A: 'a, D> {
    pub inner: Baseiter<'a, A, D>,
}

/// An iterator over the elements of an array (mutable).
///
/// Iterator element type is `&'a mut A`.
///
/// See [`.iter_mut()`](../struct.ArrayBase.html#method.iter_mut) for more information.
pub struct IterMut<'a, A: 'a, D> {
    inner: ElementsRepr<SliceIterMut<'a, A>, ElementsBaseMut<'a, A, D>>,
}

/// An iterator over the elements of an array.
///
/// Iterator element type is `&'a mut A`.
pub struct ElementsBaseMut<'a, A: 'a, D> {
    pub inner: Baseiter<'a, A, D>,
}


/// An iterator over the indexes and elements of an array.
///
/// See [`.indexed_iter()`](../struct.ArrayBase.html#method.indexed_iter) for more information.
#[derive(Clone)]
pub struct IndexedIter<'a, A: 'a, D>(ElementsBase<'a, A, D>);
/// An iterator over the indexes and elements of an array (mutable).
///
/// See [`.indexed_iter_mut()`](../struct.ArrayBase.html#method.indexed_iter_mut) for more information.
pub struct IndexedIterMut<'a, A: 'a, D>(ElementsBaseMut<'a, A, D>);

impl<'a, A, D> IndexedIter<'a, A, D>
    where D: Dimension
{
    pub(crate) fn new(x: ElementsBase<'a, A, D>) -> Self {
        IndexedIter(x)
    }
}

impl<'a, A, D> IndexedIterMut<'a, A, D>
    where D: Dimension
{
    pub(crate) fn new(x: ElementsBaseMut<'a, A, D>) -> Self {
        IndexedIterMut(x)
    }
}


impl<'a, A, D: Dimension> Iterator for Iter<'a, A, D> {
    type Item = &'a A;
    #[inline]
    fn next(&mut self) -> Option<&'a A> {
        either_mut!(self.inner, iter => iter.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        either!(self.inner, ref iter => iter.size_hint())
    }

    fn fold<Acc, G>(self, init: Acc, g: G) -> Acc
        where G: FnMut(Acc, Self::Item) -> Acc
    {
        either!(self.inner, iter => iter.fold(init, g))
    }
}

impl<'a, A> DoubleEndedIterator for Iter<'a, A, Ix1> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a A> {
        either_mut!(self.inner, iter => iter.next_back())
    }
}

impl<'a, A, D> ExactSizeIterator for Iter<'a, A, D>
    where D: Dimension
{
    fn len(&self) -> usize {
        either!(self.inner, ref iter => iter.len())
    }
}


impl<'a, A, D: Dimension> Iterator for IndexedIter<'a, A, D> {
    type Item = (D::Pattern, &'a A);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = match self.0.inner.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        match self.0.inner.next_ref() {
            None => None,
            Some(p) => Some((index.into_pattern(), p)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.inner.len();
        (len, Some(len))
    }
}

impl<'a, A, D> ExactSizeIterator for IndexedIter<'a, A, D>
    where D: Dimension
{
    fn len(&self) -> usize {
        self.0.inner.len()
    }
}

impl<'a, A, D: Dimension> Iterator for IterMut<'a, A, D> {
    type Item = &'a mut A;
    #[inline]
    fn next(&mut self) -> Option<&'a mut A> {
        either_mut!(self.inner, iter => iter.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        either!(self.inner, ref iter => iter.size_hint())
    }

    fn fold<Acc, G>(self, init: Acc, g: G) -> Acc
        where G: FnMut(Acc, Self::Item) -> Acc
    {
        either!(self.inner, iter => iter.fold(init, g))
    }
}

impl<'a, A> DoubleEndedIterator for IterMut<'a, A, Ix1> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut A> {
        either_mut!(self.inner, iter => iter.next_back())
    }
}

impl<'a, A, D> ExactSizeIterator for IterMut<'a, A, D>
    where D: Dimension
{
    fn len(&self) -> usize {
        either!(self.inner, ref iter => iter.len())
    }
}

impl<'a, A, D: Dimension> Iterator for ElementsBaseMut<'a, A, D> {
    type Item = &'a mut A;
    #[inline]
    fn next(&mut self) -> Option<&'a mut A> {
        self.inner.next_ref_mut()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.len();
        (len, Some(len))
    }

    fn fold<Acc, G>(self, init: Acc, mut g: G) -> Acc
        where G: FnMut(Acc, Self::Item) -> Acc
    {
        unsafe {
            self.inner.fold(init, move |acc, ptr| g(acc, &mut *ptr))
        }
    }
}

impl<'a, A> DoubleEndedIterator for ElementsBaseMut<'a, A, Ix1> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut A> {
        self.inner.next_back_ref_mut()
    }
}

impl<'a, A, D> ExactSizeIterator for ElementsBaseMut<'a, A, D>
    where D: Dimension
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}


impl<'a, A, D: Dimension> Iterator for IndexedIterMut<'a, A, D> {
    type Item = (D::Pattern, &'a mut A);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = match self.0.inner.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        match self.0.inner.next_ref_mut() {
            None => None,
            Some(p) => Some((index.into_pattern(), p)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.inner.len();
        (len, Some(len))
    }
}

impl<'a, A, D> ExactSizeIterator for IndexedIterMut<'a, A, D>
    where D: Dimension
{
    fn len(&self) -> usize {
        self.0.inner.len()
    }
}

/// An iterator that traverses over all dimensions but the innermost,
/// and yields each inner row.
///
/// See [`.lanes()`](../struct.ArrayBase.html#method.lanes) for more information.
pub struct LanesIter<'a, A: 'a, D> {
    inner_len: Ix,
    inner_stride: Ixs,
    iter: Baseiter<'a, A, D>,
}

impl<'a, A, D> Iterator for LanesIter<'a, A, D>
    where D: Dimension
{
    type Item = ArrayView<'a, A, Ix1>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ptr| {
            unsafe { ArrayView::new_(ptr, Ix1(self.inner_len), Ix1(self.inner_stride as Ix)) }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.iter.len();
        (len, Some(len))
    }
}

impl<'a, A, D> ExactSizeIterator for LanesIter<'a, A, D>
    where D: Dimension
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

// NOTE: LanesIterMut is a mutable iterator and must not expose aliasing
// pointers. Due to this we use an empty slice for the raw data (it's unused
// anyway).
/// An iterator that traverses over all dimensions but the innermost,
/// and yields each inner row (mutable).
///
/// See [`.lanes_mut()`](../struct.ArrayBase.html#method.lanes_mut)
/// for more information.
pub struct LanesIterMut<'a, A: 'a, D> {
    inner_len: Ix,
    inner_stride: Ixs,
    iter: Baseiter<'a, A, D>,
}

impl<'a, A, D> Iterator for LanesIterMut<'a, A, D>
    where D: Dimension,
{
    type Item = ArrayViewMut<'a, A, Ix1>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ptr| {
            unsafe {
                ArrayViewMut::new_(ptr, Ix1(self.inner_len), Ix1(self.inner_stride as Ix))
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.iter.len();
        (len, Some(len))
    }
}

impl<'a, A, D> ExactSizeIterator for LanesIterMut<'a, A, D>
    where D: Dimension,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

#[derive(Debug)]
pub struct OuterIterCore<A, D> {
    index: Ix,
    len: Ix,
    stride: Ixs,
    inner_dim: D,
    inner_strides: D,
    ptr: *mut A,
}

clone_bounds!(
    [A, D: Clone]
    OuterIterCore[A, D] {
        @copy {
            index,
            len,
            stride,
            ptr,
        }
        inner_dim,
        inner_strides,
    }
);

fn new_outer_core<A, S, D>(v: ArrayBase<S, D>, axis: usize)
    -> OuterIterCore<A, D::Smaller>
    where D: RemoveAxis,
          S: Data<Elem = A>
{
    let shape = v.shape()[axis];
    let stride = v.strides()[axis];

    OuterIterCore {
        index: 0,
        len: shape,
        stride: stride,
        inner_dim: v.dim.remove_axis(Axis(axis)),
        inner_strides: v.strides.remove_axis(Axis(axis)),
        ptr: v.ptr,
    }
}

impl<A, D> OuterIterCore<A, D> {
    unsafe fn offset(&self, index: usize) -> *mut A {
        debug_assert!(index <= self.len,
                      "index={}, len={}, stride={}", index, self.len, self.stride);
        self.ptr.offset(index as isize * self.stride)
    }
}

impl<A, D> Iterator for OuterIterCore<A, D>
    where D: Dimension,
{
    type Item = *mut A;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            None
        } else {
            let ptr = unsafe { self.offset(self.index) };
            self.index += 1;
            Some(ptr)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len - self.index;
        (len, Some(len))
    }
}

impl<A, D> DoubleEndedIterator for OuterIterCore<A, D>
    where D: Dimension,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            None
        } else {
            self.len -= 1;
            let ptr = unsafe { self.offset(self.len) };
            Some(ptr)
        }
    }
}

/// An iterator that traverses over an axis and
/// and yields each subview.
///
/// The outermost dimension is `Axis(0)`, created with `.outer_iter()`,
/// but you can traverse arbitrary dimension with `.axis_iter()`.
///
/// For example, in a 3 × 5 × 5 array, with `axis` equal to `Axis(2)`,
/// the iterator element is a 3 × 5 subview (and there are 5 in total).
///
/// Iterator element type is `ArrayView<'a, A, D>`.
///
/// See [`.outer_iter()`](../struct.ArrayBase.html#method.outer_iter)
/// or [`.axis_iter()`](../struct.ArrayBase.html#method.axis_iter)
/// for more information.
#[derive(Debug)]
pub struct AxisIter<'a, A: 'a, D> {
    iter: OuterIterCore<A, D>,
    life: PhantomData<&'a A>,
}

clone_bounds!(
    ['a, A, D: Clone]
    AxisIter['a, A, D] {
        @copy {
            life,
        }
        iter,
    }
);


macro_rules! outer_iter_split_at_impl {
    ($iter: ident) => (
        impl<'a, A, D> $iter<'a, A, D>
            where D: Dimension
        {
            /// Split the iterator at index, yielding two disjoint iterators.
            ///
            /// *panics* if `index` is strictly greater than the iterator's length
            pub fn split_at(self, index: Ix)
                -> ($iter<'a, A, D>, $iter<'a, A, D>)
            {
                assert!(index <= self.iter.len);
                let right_ptr = if index != self.iter.len {
                    unsafe { self.iter.offset(index) } 
                }
                else {
                    self.iter.ptr
                };
                let left = $iter {
                    iter: OuterIterCore {
                        index: 0,
                        len: index,
                        stride: self.iter.stride,
                        inner_dim: self.iter.inner_dim.clone(),
                        inner_strides: self.iter.inner_strides.clone(),
                        ptr: self.iter.ptr,
                    },
                    life: self.life,
                };
                let right = $iter {
                    iter: OuterIterCore {
                        index: 0,
                        len: self.iter.len - index,
                        stride: self.iter.stride,
                        inner_dim: self.iter.inner_dim,
                        inner_strides: self.iter.inner_strides,
                        ptr: right_ptr,
                    },
                    life: self.life,
                };
                (left, right)
            }
        }
    )
}

outer_iter_split_at_impl!(AxisIter);

impl<'a, A, D> Iterator for AxisIter<'a, A, D>
    where D: Dimension
{
    type Item = ArrayView<'a, A, D>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ptr| {
            unsafe {
                self.as_ref(ptr)
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, A, D> DoubleEndedIterator for AxisIter<'a, A, D>
    where D: Dimension
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|ptr| {
            unsafe {
                self.as_ref(ptr)
            }
        })
    }
}

impl<'a, A, D> ExactSizeIterator for AxisIter<'a, A, D>
    where D: Dimension
{
    fn len(&self) -> usize {
        self.size_hint().0
    }
}

pub fn new_outer_iter<A, D>(v: ArrayView<A, D>) -> AxisIter<A, D::Smaller>
    where D: RemoveAxis
{
    AxisIter {
        iter: new_outer_core(v, 0),
        life: PhantomData,
    }
}

pub fn new_axis_iter<A, D>(v: ArrayView<A, D>, axis: usize)
    -> AxisIter<A, D::Smaller>
    where D: RemoveAxis
{
    AxisIter {
        iter: new_outer_core(v, axis),
        life: PhantomData,
    }
}


/// An iterator that traverses over an axis and
/// and yields each subview (mutable)
///
/// The outermost dimension is `Axis(0)`, created with `.outer_iter()`,
/// but you can traverse arbitrary dimension with `.axis_iter()`.
///
/// For example, in a 3 × 5 × 5 array, with `axis` equal to `Axis(2)`,
/// the iterator element is a 3 × 5 subview (and there are 5 in total).
///
/// Iterator element type is `ArrayViewMut<'a, A, D>`.
///
/// See [`.outer_iter_mut()`](../struct.ArrayBase.html#method.outer_iter_mut)
/// or [`.axis_iter_mut()`](../struct.ArrayBase.html#method.axis_iter_mut)
/// for more information.
pub struct AxisIterMut<'a, A: 'a, D> {
    iter: OuterIterCore<A, D>,
    life: PhantomData<&'a mut A>,
}

outer_iter_split_at_impl!(AxisIterMut);

impl<'a, A, D> Iterator for AxisIterMut<'a, A, D>
    where D: Dimension
{
    type Item = ArrayViewMut<'a, A, D>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ptr| {
            unsafe {
                self.as_ref(ptr)
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, A, D> DoubleEndedIterator for AxisIterMut<'a, A, D>
    where D: Dimension
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|ptr| {
            unsafe {
                self.as_ref(ptr)
            }
        })
    }
}

impl<'a, A, D> ExactSizeIterator for AxisIterMut<'a, A, D>
    where D: Dimension
{
    fn len(&self) -> usize {
        self.size_hint().0
    }
}

pub fn new_outer_iter_mut<A, D>(v: ArrayViewMut<A, D>) -> AxisIterMut<A, D::Smaller>
    where D: RemoveAxis
{
    AxisIterMut {
        iter: new_outer_core(v, 0),
        life: PhantomData,
    }
}

pub fn new_axis_iter_mut<A, D>(v: ArrayViewMut<A, D>, axis: usize)
    -> AxisIterMut<A, D::Smaller>
    where D: RemoveAxis
{
    AxisIterMut {
        iter: new_outer_core(v, axis),
        life: PhantomData,
    }
}

impl<'a, A, D: Dimension> NdProducer for AxisIter<'a, A, D>
{
    type Item = <Self as Iterator>::Item;
    type Dim = Ix1;
    type Ptr = *mut A;
    type Stride = isize;

    #[doc(hidden)]
    fn layout(&self) -> ::Layout {
        ::Layout::one_dimensional()
    }
    #[doc(hidden)]
    fn raw_dim(&self) -> Self::Dim {
        Ix1(self.len())
    }
    #[doc(hidden)]
    fn as_ptr(&self) -> Self::Ptr {
        self.iter.ptr
    }

    fn contiguous_stride(&self) -> isize {
        self.iter.stride
    }

    #[doc(hidden)]
    unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
        ArrayView::new_(ptr,
                        self.iter.inner_dim.clone(),
                        self.iter.inner_strides.clone())
    }
    #[doc(hidden)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
        self.iter.ptr.offset(self.iter.stride * i[0] as isize)
    }

    #[doc(hidden)]
    fn stride_of(&self, _axis: Axis) -> isize {
        self.contiguous_stride()
    }

    #[doc(hidden)]
    fn split_at(self, _axis: Axis, index: usize) -> (Self, Self) {
        self.split_at(index)
    }
    private_impl!{}
}

impl<'a, A, D: Dimension> NdProducer for AxisIterMut<'a, A, D>
{
    type Item = <Self as Iterator>::Item;
    type Dim = Ix1;
    type Ptr = *mut A;
    type Stride = isize;

    #[doc(hidden)]
    fn layout(&self) -> ::Layout {
        ::Layout::one_dimensional()
    }
    #[doc(hidden)]
    fn raw_dim(&self) -> Self::Dim {
        Ix1(self.len())
    }
    #[doc(hidden)]
    fn as_ptr(&self) -> Self::Ptr {
        self.iter.ptr
    }

    fn contiguous_stride(&self) -> isize {
        self.iter.stride
    }

    #[doc(hidden)]
    unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
        ArrayViewMut::new_(ptr,
                           self.iter.inner_dim.clone(),
                           self.iter.inner_strides.clone())
    }
    #[doc(hidden)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
        self.iter.ptr.offset(self.iter.stride * i[0] as isize)
    }

    #[doc(hidden)]
    fn stride_of(&self, _axis: Axis) -> isize {
        self.contiguous_stride()
    }

    #[doc(hidden)]
    fn split_at(self, _axis: Axis, index: usize) -> (Self, Self) {
        self.split_at(index)
    }
    private_impl!{}
}

/// An iterator that traverses over the specified axis
/// and yields views of the specified size on this axis.
///
/// For example, in a 2 × 8 × 3 array, if the axis of iteration
/// is 1 and the chunk size is 2, the yielded elements
/// are 2 × 2 × 3 views (and there are 4 in total).
///
/// Iterator element type is `ArrayView<'a, A, D>`.
///
/// See [`.axis_chunks_iter()`](../struct.ArrayBase.html#method.axis_chunks_iter) for more information.
pub struct AxisChunksIter<'a, A: 'a, D> {
    iter: OuterIterCore<A, D>,
    last_ptr: *mut A,
    last_dim: D,
    life: PhantomData<&'a A>,
}

clone_bounds!(
    ['a, A, D: Clone]
    AxisChunksIter['a, A, D] {
        @copy {
            life,
            last_ptr,
        }
        iter,
        last_dim,
    }
);

fn chunk_iter_parts<A, D: Dimension>(v: ArrayView<A, D>, axis: usize, size: usize)
    -> (OuterIterCore<A, D>, *mut A, D)
{
    let axis_len = v.shape()[axis];
    let size = if size > axis_len { axis_len } else { size };
    let last_index = axis_len / size;
    let rem = axis_len % size;
    let shape = if rem == 0 { last_index } else { last_index + 1 };
    let stride = v.strides()[axis] * size as isize;

    let mut inner_dim = v.dim.clone();
    inner_dim.slice_mut()[axis] = size;

    let mut last_dim = v.dim;
    last_dim.slice_mut()[axis] = if rem == 0 { size } else { rem };

    let last_ptr = if rem != 0 {
        unsafe {
            v.ptr.offset(stride * last_index as isize)
        }
    }
    else {
        v.ptr
    };
    let iter = OuterIterCore {
        index: 0,
        len: shape,
        stride: stride,
        inner_dim: inner_dim,
        inner_strides: v.strides,
        ptr: v.ptr,
    };

    (iter, last_ptr, last_dim)
}

pub fn new_chunk_iter<A, D>(v: ArrayView<A, D>, axis: usize, size: usize)
    -> AxisChunksIter<A, D>
    where D: Dimension
{
    let (iter, last_ptr, last_dim) = chunk_iter_parts(v, axis, size);

    AxisChunksIter {
        iter: iter,
        last_ptr: last_ptr,
        last_dim: last_dim,
        life: PhantomData,
    }
}

macro_rules! chunk_iter_impl {
    ($iter:ident, $array:ident) => (
        impl<'a, A, D> $iter<'a, A, D>
            where D: Dimension
        {
            fn get_subview(&self, iter_item: Option<*mut A>)
                -> Option<$array<'a, A, D>>
            {
                iter_item.map(|ptr| {
                    if ptr != self.last_ptr {
                        unsafe {
                            $array::new_(ptr,
                                         self.iter.inner_dim.clone(),
                                         self.iter.inner_strides.clone())
                        }
                    }
                    else {
                        unsafe {
                            $array::new_(ptr,
                                         self.last_dim.clone(),
                                         self.iter.inner_strides.clone())
                        }
                    }
                })
            }
        }

        impl<'a, A, D> Iterator for $iter<'a, A, D>
            where D: Dimension,
        {
            type Item = $array<'a, A, D>;

            fn next(&mut self) -> Option<Self::Item> {
                let res = self.iter.next();
                self.get_subview(res)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.iter.size_hint()
            }
        }

        impl<'a, A, D> DoubleEndedIterator for $iter<'a, A, D>
            where D: Dimension,
        {
            fn next_back(&mut self) -> Option<Self::Item> {
                let res = self.iter.next_back();
                self.get_subview(res)
            }
        }

        impl<'a, A, D> ExactSizeIterator for $iter<'a, A, D>
            where D: Dimension,
        { }
    )
}

/// An iterator that traverses over the specified axis
/// and yields mutable views of the specified size on this axis.
///
/// For example, in a 2 × 8 × 3 array, if the axis of iteration
/// is 1 and the chunk size is 2, the yielded elements
/// are 2 × 2 × 3 views (and there are 4 in total).
///
/// Iterator element type is `ArrayViewMut<'a, A, D>`.
///
/// See [`.axis_chunks_iter_mut()`](../struct.ArrayBase.html#method.axis_chunks_iter_mut)
/// for more information.
pub struct AxisChunksIterMut<'a, A: 'a, D> {
    iter: OuterIterCore<A, D>,
    last_ptr: *mut A,
    last_dim: D,
    life: PhantomData<&'a mut A>,
}

pub fn new_chunk_iter_mut<A, D>(v: ArrayViewMut<A, D>, axis: usize, size: usize)
    -> AxisChunksIterMut<A, D>
    where D: Dimension
{
    let (iter, last_ptr, last_dim) = chunk_iter_parts(v.into_view(), axis, size);

    AxisChunksIterMut {
        iter: iter,
        last_ptr: last_ptr,
        last_dim: last_dim,
        life: PhantomData,
    }
}

chunk_iter_impl!(AxisChunksIter, ArrayView);
chunk_iter_impl!(AxisChunksIterMut, ArrayViewMut);


send_sync_read_only!(Iter);
send_sync_read_only!(IndexedIter);
send_sync_read_only!(LanesIter);
send_sync_read_only!(AxisIter);
send_sync_read_only!(AxisChunksIter);
send_sync_read_only!(ElementsBase);

send_sync_read_write!(IterMut);
send_sync_read_write!(IndexedIterMut);
send_sync_read_write!(LanesIterMut);
send_sync_read_write!(AxisIterMut);
send_sync_read_write!(AxisChunksIterMut);
send_sync_read_write!(ElementsBaseMut);

/// (Trait used internally) An iterator that we trust
/// to deliver exactly as many items as it said it would.
pub unsafe trait TrustedIterator { }

use std;
use linspace::Linspace;
use iter::IndicesIter;
use indexes::IndicesIterF;

unsafe impl<F> TrustedIterator for Linspace<F> { }
unsafe impl<'a, A, D> TrustedIterator for Iter<'a, A, D> { }
unsafe impl<I, F> TrustedIterator for std::iter::Map<I, F>
    where I: TrustedIterator { }
unsafe impl<'a, A> TrustedIterator for slice::Iter<'a, A> { }
unsafe impl TrustedIterator for ::std::ops::Range<usize> { }
// FIXME: These indices iter are dubious -- size needs to be checked up front.
unsafe impl<D> TrustedIterator for IndicesIter<D> where D: Dimension { }
unsafe impl<D> TrustedIterator for IndicesIterF<D> where D: Dimension { }


/// Like Iterator::collect, but only for trusted length iterators
pub fn to_vec<I>(iter: I) -> Vec<I::Item>
    where I: TrustedIterator + ExactSizeIterator
{
    to_vec_mapped(iter, |x| x)
}

/// Like Iterator::collect, but only for trusted length iterators
pub fn to_vec_mapped<I, F, B>(iter: I, mut f: F) -> Vec<B>
    where I: TrustedIterator + ExactSizeIterator,
          F: FnMut(I::Item) -> B,
{
    // Use an `unsafe` block to do this efficiently.
    // We know that iter will produce exactly .size() elements,
    // and the loop can vectorize if it's clean (without branch to grow the vector).
    let (size, _) = iter.size_hint();
    let mut result = Vec::with_capacity(size);
    let mut out_ptr = result.as_mut_ptr();
    let mut len = 0;
    iter.fold((), |(), elt| {
        unsafe {
            ptr::write(out_ptr, f(elt));
            len += 1;
            result.set_len(len);
            out_ptr = out_ptr.offset(1);
        }
    });
    debug_assert_eq!(size, result.len());
    result
}
