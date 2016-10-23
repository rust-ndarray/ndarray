// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use std::marker::PhantomData;
use std::ptr;

use Ix1;

use super::{Dimension, Ix, Ixs};
use super::{Elements, ElementsRepr, ElementsBase, ElementsBaseMut, ElementsMut, Indexed, IndexedMut};
use super::{
    ArrayBase,
    Data,
    ArrayView,
    ArrayViewMut,
    RemoveAxis,
    Axis,
};

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

    fn size_hint(&self) -> usize {
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
        assert!(ndim > 0);
        let mut accum = init;
        loop {
            if let Some(mut index) = self.index.clone() {
                let stride = self.strides.last_elem() as isize;
                let elem_index = index.last_elem();
                let len = self.dim.last_elem();
                let offset = D::stride_offset(&index, &self.strides);
                for i in 0..len - elem_index {
                    unsafe {
                        accum = g(accum, self.ptr.offset(offset + i as isize * stride));
                    }
                }
                index.slice_mut()[ndim - 1] = len - 1;
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
        self.dim -= 1;
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

impl<'a, A, D: Clone> Clone for Baseiter<'a, A, D> {
    fn clone(&self) -> Baseiter<'a, A, D> {
        Baseiter {
            ptr: self.ptr,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            index: self.index.clone(),
            life: self.life,
        }
    }
}

impl<'a, A, D: Clone> Clone for ElementsBase<'a, A, D> {
    fn clone(&self) -> ElementsBase<'a, A, D> {
        ElementsBase { inner: self.inner.clone() }
    }
}

impl<'a, A, D: Dimension> Iterator for ElementsBase<'a, A, D> {
    type Item = &'a A;
    #[inline]
    fn next(&mut self) -> Option<&'a A> {
        self.inner.next_ref()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.size_hint();
        (len, Some(len))
    }

    fn fold<Acc, G>(self, init: Acc, mut g: G) -> Acc
        where G: FnMut(Acc, Self::Item) -> Acc,
    {
        unsafe {
            self.inner.fold(init, |acc, ptr| g(acc, &*ptr))
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
{}

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


impl<'a, A, D: Clone> Clone for Elements<'a, A, D> {
    fn clone(&self) -> Elements<'a, A, D> {
        Elements {
            inner: match self.inner {
                ElementsRepr::Slice(ref iter) => ElementsRepr::Slice(iter.clone()),
                ElementsRepr::Counted(ref iter) => {
                    ElementsRepr::Counted(iter.clone())
                }
            },
        }
    }
}

impl<'a, A, D: Dimension> Iterator for Elements<'a, A, D> {
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

impl<'a, A> DoubleEndedIterator for Elements<'a, A, Ix1> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a A> {
        either_mut!(self.inner, iter => iter.next_back())
    }
}

impl<'a, A, D> ExactSizeIterator for Elements<'a, A, D>
    where D: Dimension
{}


impl<'a, A, D: Dimension> Iterator for Indexed<'a, A, D> {
    type Item = (D, &'a A);
    #[inline]
    fn next(&mut self) -> Option<(D, &'a A)> {
        let index = match self.0.inner.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        match self.0.inner.next_ref() {
            None => None,
            Some(p) => Some((index, p)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.inner.size_hint();
        (len, Some(len))
    }
}

impl<'a, A, D: Dimension> Iterator for ElementsMut<'a, A, D> {
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

impl<'a, A> DoubleEndedIterator for ElementsMut<'a, A, Ix1> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut A> {
        either_mut!(self.inner, iter => iter.next_back())
    }
}

impl<'a, A, D> ExactSizeIterator for ElementsMut<'a, A, D>
    where D: Dimension
{}

impl<'a, A, D: Dimension> Iterator for ElementsBaseMut<'a, A, D> {
    type Item = &'a mut A;
    #[inline]
    fn next(&mut self) -> Option<&'a mut A> {
        self.inner.next_ref_mut()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.size_hint();
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

impl<'a, A, D: Dimension> Iterator for IndexedMut<'a, A, D> {
    type Item = (D, &'a mut A);
    #[inline]
    fn next(&mut self) -> Option<(D, &'a mut A)> {
        let index = match self.0.inner.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        match self.0.inner.next_ref_mut() {
            None => None,
            Some(p) => Some((index, p)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.inner.size_hint();
        (len, Some(len))
    }
}

/// An iterator that traverses over all dimensions but the innermost,
/// and yields each inner row.
///
/// See [`.inner_iter()`](struct.ArrayBase.html#method.inner_iter) for more information.
pub struct InnerIter<'a, A: 'a, D> {
    inner_len: Ix,
    inner_stride: Ixs,
    iter: Baseiter<'a, A, D>,
}

pub fn new_inner_iter<A, D>(mut v: ArrayView<A, D>) -> InnerIter<A, D>
    where D: Dimension
{
    if v.shape().len() == 0 {
        InnerIter {
            inner_len: 1,
            inner_stride: 1,
            iter: v.into_base_iter(),
        }
    } else {
        // Set length of innerest dimension to 1, start iteration
        let ndim = v.shape().len();
        let len = v.shape()[ndim - 1];
        let stride = v.strides()[ndim - 1];
        v.dim.slice_mut()[ndim - 1] = 1;
        InnerIter {
            inner_len: len,
            inner_stride: stride,
            iter: v.into_base_iter(),
        }
    }
}

impl<'a, A, D> Iterator for InnerIter<'a, A, D>
    where D: Dimension
{
    type Item = ArrayView<'a, A, Ix1>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ptr| {
            unsafe { ArrayView::new_(ptr, self.inner_len, self.inner_stride as Ix) }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.iter.size_hint();
        (len, Some(len))
    }
}

impl<'a, A, D> ExactSizeIterator for InnerIter<'a, A, D>
    where D: Dimension
{}

// NOTE: InnerIterMut is a mutable iterator and must not expose aliasing
// pointers. Due to this we use an empty slice for the raw data (it's unused
// anyway).
/// An iterator that traverses over all dimensions but the innermost,
/// and yields each inner row (mutable).
///
/// See [`.inner_iter_mut()`](struct.ArrayBase.html#method.inner_iter_mut)
/// for more information.
pub struct InnerIterMut<'a, A: 'a, D> {
    inner_len: Ix,
    inner_stride: Ixs,
    iter: Baseiter<'a, A, D>,
}

pub fn new_inner_iter_mut<A, D>(mut v: ArrayViewMut<A, D>) -> InnerIterMut<A, D>
    where D: Dimension,
{
    if v.shape().len() == 0 {
        InnerIterMut {
            inner_len: 1,
            inner_stride: 1,
            iter: v.into_base_iter(),
        }
    } else {
        // Set length of innerest dimension to 1, start iteration
        let ndim = v.shape().len();
        let len = v.shape()[ndim - 1];
        let stride = v.strides()[ndim - 1];
        v.dim.slice_mut()[ndim - 1] = 1;
        InnerIterMut {
            inner_len: len,
            inner_stride: stride,
            iter: v.into_base_iter(),
        }
    }
}

impl<'a, A, D> Iterator for InnerIterMut<'a, A, D>
    where D: Dimension,
{
    type Item = ArrayViewMut<'a, A, Ix1>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ptr| {
            unsafe {
                ArrayViewMut::new_(ptr, self.inner_len, self.inner_stride as Ix)
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.iter.size_hint();
        (len, Some(len))
    }
}

impl<'a, A, D> ExactSizeIterator for InnerIterMut<'a, A, D>
    where D: Dimension,
{ }

pub struct OuterIterCore<A, D> {
    index: Ix,
    len: Ix,
    stride: Ixs,
    inner_dim: D,
    inner_strides: D,
    ptr: *mut A,
}

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
/// See [`.outer_iter()`](struct.ArrayBase.html#method.outer_iter)
/// or [`.axis_iter()`](struct.ArrayBase.html#method.axis_iter)
/// for more information.
pub struct AxisIter<'a, A: 'a, D> {
    iter: OuterIterCore<A, D>,
    life: PhantomData<&'a A>,
}

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

impl<'a, A, D> Clone for AxisIter<'a, A, D>
    where D: Dimension
{
    fn clone(&self) -> Self {
        AxisIter {
            iter: OuterIterCore {
                index: self.iter.index,
                len: self.iter.len,
                stride: self.iter.stride,
                inner_dim: self.iter.inner_dim.clone(),
                inner_strides: self.iter.inner_strides.clone(),
                ptr: self.iter.ptr,
            },
            life: self.life,
        }
    }
}

impl<'a, A, D> Iterator for AxisIter<'a, A, D>
    where D: Dimension
{
    type Item = ArrayView<'a, A, D>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ptr| {
            unsafe {
                ArrayView::new_(ptr,
                                self.iter.inner_dim.clone(),
                                self.iter.inner_strides.clone())
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
                ArrayView::new_(ptr,
                                self.iter.inner_dim.clone(),
                                self.iter.inner_strides.clone())
            }
        })
    }
}

impl<'a, A, D> ExactSizeIterator for AxisIter<'a, A, D>
    where D: Dimension
{}

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
/// See [`.outer_iter_mut()`](struct.ArrayBase.html#method.outer_iter_mut)
/// or [`.axis_iter_mut()`](struct.ArrayBase.html#method.axis_iter_mut)
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
                ArrayViewMut::new_(ptr,
                                   self.iter.inner_dim.clone(),
                                   self.iter.inner_strides.clone())
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
                ArrayViewMut::new_(ptr,
                                   self.iter.inner_dim.clone(),
                                   self.iter.inner_strides.clone())
            }
        })
    }
}

impl<'a, A, D> ExactSizeIterator for AxisIterMut<'a, A, D>
    where D: Dimension
{}

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

/// An iterator that traverses over the specified axis
/// and yields views of the specified size on this axis.
///
/// For example, in a 2 × 8 × 3 array, if the axis of iteration
/// is 1 and the chunk size is 2, the yielded elements
/// are 2 × 2 × 3 views (and there are 4 in total).
///
/// Iterator element type is `ArrayView<'a, A, D>`.
///
/// See [`.axis_chunks_iter()`](struct.ArrayBase.html#method.axis_chunks_iter) for more information.
pub struct AxisChunksIter<'a, A: 'a, D> {
    iter: OuterIterCore<A, D>,
    last_ptr: *mut A,
    last_dim: D,
    life: PhantomData<&'a A>,
}

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

    let mut last_dim = v.dim.clone();
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
        inner_strides: v.strides.clone(),
        ptr: v.ptr,
    };

    (iter, last_ptr, last_dim)
}

pub fn new_chunk_iter<A, D>(v: ArrayView<A, D>, axis: usize, size: usize)
    -> AxisChunksIter<A, D>
    where D: Dimension
{
    let (iter, last_ptr, last_dim) = chunk_iter_parts(v.view(), axis, size);

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
/// See [`.axis_chunks_iter_mut()`](struct.ArrayBase.html#method.axis_chunks_iter_mut)
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
    let (iter, last_ptr, last_dim) = chunk_iter_parts(v.view(), axis, size);

    AxisChunksIterMut {
        iter: iter,
        last_ptr: last_ptr,
        last_dim: last_dim,
        life: PhantomData,
    }
}

chunk_iter_impl!(AxisChunksIter, ArrayView);
chunk_iter_impl!(AxisChunksIterMut, ArrayViewMut);


// Send and Sync
// All the iterators are thread safe the same way the slice's iterator are

// read-only iterators use Sync => Send rules, same as `std::slice::Iter`.
macro_rules! send_sync_read_only {
    ($name:ident) => {
        unsafe impl<'a, A, D> Send for $name<'a, A, D> where A: Sync, D: Send { }
        unsafe impl<'a, A, D> Sync for $name<'a, A, D> where A: Sync, D: Sync { }
    }
}

// read-write iterators use Send => Send rules, same as `std::slice::IterMut`.
macro_rules! send_sync_read_write {
    ($name:ident) => {
        unsafe impl<'a, A, D> Send for $name<'a, A, D> where A: Send, D: Send { }
        unsafe impl<'a, A, D> Sync for $name<'a, A, D> where A: Sync, D: Sync { }
    }
}

send_sync_read_only!(Elements);
send_sync_read_only!(Indexed);
send_sync_read_only!(InnerIter);
send_sync_read_only!(AxisIter);
send_sync_read_only!(AxisChunksIter);

send_sync_read_write!(ElementsMut);
send_sync_read_write!(IndexedMut);
send_sync_read_write!(InnerIterMut);
send_sync_read_write!(AxisIterMut);
send_sync_read_write!(AxisChunksIterMut);

/// (Trait used internally) An iterator that we trust
/// to deliver exactly as many items as it said it would.
pub unsafe trait TrustedIterator { }

use std::slice;
use std::iter;
use linspace::Linspace;
use indexes::Indexes;

unsafe impl<F> TrustedIterator for Linspace<F> { }
unsafe impl<'a, A, D> TrustedIterator for Elements<'a, A, D> { }
unsafe impl<I, F> TrustedIterator for iter::Map<I, F>
    where I: TrustedIterator { }
unsafe impl<'a, A> TrustedIterator for slice::Iter<'a, A> { }
unsafe impl TrustedIterator for ::std::ops::Range<usize> { }
unsafe impl<D> TrustedIterator for Indexes<D> where D: Dimension { }


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
