use std::marker;

use super::{Dimension, Ix};
use super::{to_ref, to_ref_mut};
use super::{Elements, ElementsMut, IndexedElements, IndexedElementsMut};

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
    fn next(&mut self) -> Option<*mut A>
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
        unsafe { self.next().map(|p| to_ref(p as *const _)) }
    }

    #[inline]
    fn next_ref_mut(&mut self) -> Option<&'a mut A>
    {
        unsafe { self.next().map(|p| to_ref_mut(p)) }
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
        unsafe { self.next_back().map(|p| to_ref(p as *const _)) }
    }

    #[inline]
    fn next_back_ref_mut(&mut self) -> Option<&'a mut A>
    {
        unsafe { self.next_back().map(|p| to_ref_mut(p)) }
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

impl<'a, A, D: Clone> Clone for Elements<'a, A, D>
{
    fn clone(&self) -> Elements<'a, A, D> { Elements{inner: self.inner.clone()} }
}

impl<'a, A, D: Dimension> Iterator for Elements<'a, A, D>
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

impl<'a, A> DoubleEndedIterator for Elements<'a, A, Ix>
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a A>
    {
        self.inner.next_back_ref()
    }
}

impl<'a, A> ExactSizeIterator for Elements<'a, A, Ix> { }

impl<'a, A, D: Clone> Clone for IndexedElements<'a, A, D>
{
    fn clone(&self) -> IndexedElements<'a, A, D> {
        IndexedElements{inner: self.inner.clone()}
    }
}

impl<'a, A, D: Dimension> Iterator for IndexedElements<'a, A, D>
{
    type Item = (D, &'a A);
    #[inline]
    fn next(&mut self) -> Option<(D, &'a A)>
    {
        let index = match self.inner.index {
            None => return None,
            Some(ref ix) => ix.clone()
        };
        match self.inner.next_ref() {
            None => None,
            Some(p) => Some((index, p))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>)
    {
        let len = self.inner.size_hint();
        (len, Some(len))
    }
}

impl<'a, A, D: Dimension> Iterator for ElementsMut<'a, A, D>
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

impl<'a, A> DoubleEndedIterator for ElementsMut<'a, A, Ix>
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut A>
    {
        self.inner.next_back_ref_mut()
    }
}

impl<'a, A, D: Dimension> Iterator for IndexedElementsMut<'a, A, D>
{
    type Item = (D, &'a mut A);
    #[inline]
    fn next(&mut self) -> Option<(D, &'a mut A)>
    {
        let index = match self.inner.index {
            None => return None,
            Some(ref ix) => ix.clone()
        };
        match self.inner.next_ref_mut() {
            None => None,
            Some(p) => Some((index, p))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>)
    {
        let len = self.inner.size_hint();
        (len, Some(len))
    }
}

