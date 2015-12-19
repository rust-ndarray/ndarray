use std::marker;

use super::{Dimension, Ix, Ixs};
use super::{Elements, ElementsRepr, ElementsBase, ElementsBaseMut, ElementsMut, Indexed, IndexedMut};
use super::{
    ArrayView,
    ArrayViewMut,
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
    pub life: marker::PhantomData<&'a A>,
}


impl<'a, A, D: Dimension> Baseiter<'a, A, D>
{
    /// Creating a Baseiter is unsafe, because it can
    /// have any lifetime, be immut or mut, and the
    /// boundary and stride parameters need to be correct to
    /// avoid memory unsafety.
    ///
    /// It must be placed in the correct mother iterator to be safe.
    ///
    /// NOTE: Mind the lifetime, it's arbitrary
    #[inline]
    pub unsafe fn new(ptr: *mut A, len: D, stride: D) -> Baseiter<'a, A, D>
    {
        Baseiter {
            ptr: ptr,
            index: len.first_index(),
            dim: len,
            strides: stride,
            life: marker::PhantomData,
        }
    }
}

impl<'a, A, D: Dimension> Baseiter<'a, A, D>
{
    #[inline]
    pub fn next(&mut self) -> Option<*mut A>
    {
        let index = match self.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        let offset = Dimension::stride_offset(&index, &self.strides);
        self.index = self.dim.next_for(index);
        unsafe {
            Some(self.ptr.offset(offset))
        }
    }

    #[inline]
    fn next_ref(&mut self) -> Option<&'a A>
    {
        unsafe { self.next().map(|p| &*p) }
    }

    #[inline]
    fn next_ref_mut(&mut self) -> Option<&'a mut A>
    {
        unsafe { self.next().map(|p| &mut *p) }
    }

    fn size_hint(&self) -> usize
    {
        match self.index {
            None => 0,
            Some(ref ix) => {
                let gone = self.dim.default_strides().slice().iter()
                            .zip(ix.slice().iter())
                                 .fold(0, |s, (&a, &b)| s + a as usize * b as usize);
                self.dim.size() - gone
            }
        }
    }
}

impl<'a, A> Baseiter<'a, A, Ix>
{
    #[inline]
    fn next_back(&mut self) -> Option<*mut A>
    {
        let index = match self.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        self.dim -= 1;
        let offset = Dimension::stride_offset(&self.dim, &self.strides);
        if index == self.dim {
            self.index = None;
        }

        unsafe {
            Some(self.ptr.offset(offset))
        }
    }

    #[inline]
    fn next_back_ref(&mut self) -> Option<&'a A>
    {
        unsafe { self.next_back().map(|p| &*p) }
    }

    #[inline]
    fn next_back_ref_mut(&mut self) -> Option<&'a mut A>
    {
        unsafe { self.next_back().map(|p| &mut *p) }
    }
}

impl<'a, A, D: Clone> Clone for Baseiter<'a, A, D>
{
    fn clone(&self) -> Baseiter<'a, A, D>
    {
        Baseiter {
            ptr: self.ptr,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            index: self.index.clone(),
            life: self.life,
        }
    }
}

impl<'a, A, D: Clone> Clone for ElementsBase<'a, A, D>
{
    fn clone(&self) -> ElementsBase<'a, A, D> { ElementsBase{inner: self.inner.clone()} }
}

impl<'a, A, D: Dimension> Iterator for ElementsBase<'a, A, D>
{
    type Item = &'a A;
    #[inline]
    fn next(&mut self) -> Option<&'a A>
    {
        self.inner.next_ref()
    }

    fn size_hint(&self) -> (usize, Option<usize>)
    {
        let len = self.inner.size_hint();
        (len, Some(len))
    }
}

impl<'a, A> DoubleEndedIterator for ElementsBase<'a, A, Ix>
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a A>
    {
        self.inner.next_back_ref()
    }
}

impl<'a, A> ExactSizeIterator for ElementsBase<'a, A, Ix> { }

macro_rules! either {
    ($value:expr, $inner:ident => $result:expr) => (
        match $value {
            ElementsRepr::Slice(ref $inner) => $result,
            ElementsRepr::Counted(ref $inner) => $result,
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

impl<'a, A, D: Clone> Clone for Elements<'a, A, D>
{
    fn clone(&self) -> Elements<'a, A, D> {
        Elements {
            inner: match self.inner {
                ElementsRepr::Slice(ref iter) => ElementsRepr::Slice(iter.clone()),
                ElementsRepr::Counted(ref iter) => ElementsRepr::Counted(iter.clone()),
            }
        }
    }
}

impl<'a, A, D: Dimension> Iterator for Elements<'a, A, D>
{
    type Item = &'a A;
    #[inline]
    fn next(&mut self) -> Option<&'a A> {
        either_mut!(self.inner, iter => iter.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>)
    {
        either!(self.inner, iter => iter.size_hint())
    }
}

impl<'a, A> DoubleEndedIterator for Elements<'a, A, Ix>
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a A> {
        either_mut!(self.inner, iter => iter.next_back())
    }
}

impl<'a, A> ExactSizeIterator for Elements<'a, A, Ix> { }


impl<'a, A, D: Dimension> Iterator for Indexed<'a, A, D>
{
    type Item = (D, &'a A);
    #[inline]
    fn next(&mut self) -> Option<(D, &'a A)>
    {
        let index = match self.0.inner.index {
            None => return None,
            Some(ref ix) => ix.clone()
        };
        match self.0.inner.next_ref() {
            None => None,
            Some(p) => Some((index, p))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>)
    {
        let len = self.0.inner.size_hint();
        (len, Some(len))
    }
}

impl<'a, A, D: Dimension> Iterator for ElementsMut<'a, A, D>
{
    type Item = &'a mut A;
    #[inline]
    fn next(&mut self) -> Option<&'a mut A> {
        either_mut!(self.inner, iter => iter.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        either!(self.inner, iter => iter.size_hint())
    }
}

impl<'a, A> DoubleEndedIterator for ElementsMut<'a, A, Ix>
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut A> {
        either_mut!(self.inner, iter => iter.next_back())
    }
}

impl<'a, A, D: Dimension> Iterator for ElementsBaseMut<'a, A, D>
{
    type Item = &'a mut A;
    #[inline]
    fn next(&mut self) -> Option<&'a mut A>
    {
        self.inner.next_ref_mut()
    }

    fn size_hint(&self) -> (usize, Option<usize>)
    {
        let len = self.inner.size_hint();
        (len, Some(len))
    }
}

impl<'a, A> DoubleEndedIterator for ElementsBaseMut<'a, A, Ix>
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut A>
    {
        self.inner.next_back_ref_mut()
    }
}

impl<'a, A, D: Dimension> Iterator for IndexedMut<'a, A, D>
{
    type Item = (D, &'a mut A);
    #[inline]
    fn next(&mut self) -> Option<(D, &'a mut A)>
    {
        let index = match self.0.inner.index {
            None => return None,
            Some(ref ix) => ix.clone()
        };
        match self.0.inner.next_ref_mut() {
            None => None,
            Some(p) => Some((index, p))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>)
    {
        let len = self.0.inner.size_hint();
        (len, Some(len))
    }
}

pub struct OuterIter<'a, A: 'a, D> {
    inner_len: Ix,
    inner_stride: Ixs,
    iter: Baseiter<'a, A, D>,
}

pub fn new_outer<A, D>(mut v: ArrayView<A, D>) -> OuterIter<A, D>
    where D: Dimension,
{
    if v.shape().len() == 0 {
        OuterIter {
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
        OuterIter {
            inner_len: len,
            inner_stride: stride,
            iter: v.into_base_iter(),
        }
    }
}

impl<'a, A, D> Iterator for OuterIter<'a, A, D>
    where D: Dimension,
{
    type Item = ArrayView<'a, A, Ix>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ptr| {
            let view = ArrayView {
                data: &[],
                ptr: ptr,
                dim: self.inner_len,
                strides: self.inner_stride as Ix,
            };
            view
        })
    }
}

pub struct OuterIterMut<'a, A: 'a, D> {
    inner_len: Ix,
    inner_stride: Ixs,
    iter: Baseiter<'a, A, D>,
}

pub fn new_outer_mut<A, D>(mut v: ArrayViewMut<A, D>) -> OuterIterMut<A, D>
    where D: Dimension,
{
    if v.shape().len() == 0 {
        OuterIterMut {
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
        OuterIterMut {
            inner_len: len,
            inner_stride: stride,
            iter: v.into_base_iter(),
        }
    }
}

impl<'a, A, D> Iterator for OuterIterMut<'a, A, D>
    where D: Dimension,
{
    type Item = ArrayViewMut<'a, A, Ix>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ptr| {
            let view = ArrayViewMut {
                data: &mut [],
                ptr: ptr,
                dim: self.inner_len,
                strides: self.inner_stride as Ix,
            };
            view
        })
    }
}
