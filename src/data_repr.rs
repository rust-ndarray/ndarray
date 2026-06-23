use crate::extension::nonnull;
#[cfg(not(feature = "std"))]
use alloc::borrow::ToOwned;
use alloc::slice;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::ops::Range;
use std::mem;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

#[allow(unused_imports)]
use rawpointer::PointerExt;

/// Array's representation.
///
/// *Don’t use this type directly—use the type alias
/// [`Array`](crate::Array) for the array type!*
// Like a Vec, but with non-unique ownership semantics
//
// repr(C) to make it transmutable OwnedRepr<A> -> OwnedRepr<B> if
// transmutable A -> B.
#[derive(Debug)]
#[repr(C)]
pub struct OwnedRepr<A>
{
    ptr: NonNull<A>,
    len: usize,
    capacity: usize,
}

// OwnedRepr is a wrapper for a uniquely held allocation. Currently it is allocated by using a Vec
// (from/to raw parts) which gives the benefit that it can always be converted to/from a Vec
// cheaply.
impl<A> OwnedRepr<A>
{
    pub(crate) fn from(v: Vec<A>) -> Self
    {
        let mut v = ManuallyDrop::new(v);
        let len = v.len();
        let capacity = v.capacity();
        let ptr = nonnull::nonnull_from_vec_data(&mut v);
        Self { ptr, len, capacity }
    }

    pub(crate) fn into_vec(self) -> Vec<A>
    {
        ManuallyDrop::new(self).take_as_vec()
    }

    pub(crate) fn as_slice(&self) -> &[A]
    {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub(crate) fn len(&self) -> usize
    {
        self.len
    }

    #[cfg(test)]
    /// Note: Capacity comes from OwnedRepr (Vec)'s allocation strategy and cannot be absolutely
    /// guaranteed.
    pub(crate) fn capacity(&self) -> usize
    {
        self.capacity
    }

    pub(crate) fn as_ptr(&self) -> *const A
    {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_nonnull_mut(&mut self) -> NonNull<A>
    {
        self.ptr
    }

    /// Return end pointer
    pub(crate) fn as_end_nonnull(&self) -> NonNull<A>
    {
        unsafe { self.ptr.add(self.len) }
    }

    /// Reserve `additional` elements; return the new pointer
    ///
    /// ## Safety
    ///
    /// Note that existing pointers into the data are invalidated
    #[must_use = "must use new pointer to update existing pointers"]
    pub(crate) fn reserve(&mut self, additional: usize) -> NonNull<A>
    {
        self.modify_as_vec(|mut v| {
            v.reserve(additional);
            v
        });
        self.as_nonnull_mut()
    }

    /// Truncate "at front and back", preserve only elements inside the range,
    /// then call Vec::shrink_to_fit.
    /// Moving elements will invalidate existing pointers.
    ///
    /// Return the new lowest address pointer of the allocation.
    #[must_use = "must use new pointer to update existing pointers"]
    pub(crate) fn preserve_range_and_shrink(&mut self, span: Range<usize>) -> NonNull<A>
    {
        self.modify_as_vec(|mut v| {
            v.truncate(span.end);
            if span.start > 0 {
                v.drain(..span.start);
            }
            // Vec::shrink_to_fit is allowed to reallocate and invalidate pointers
            v.shrink_to_fit();
            v
        });
        self.as_nonnull_mut()
    }

    /// Set the valid length of the data
    ///
    /// ## Safety
    ///
    /// The first `new_len` elements of the data should be valid.
    pub(crate) unsafe fn set_len(&mut self, new_len: usize)
    {
        debug_assert!(new_len <= self.capacity);
        self.len = new_len;
    }

    /// Return the length (number of elements in total)
    pub(crate) fn release_all_elements(&mut self) -> usize
    {
        let ret = self.len;
        self.len = 0;
        ret
    }

    /// Cast self into equivalent repr of other element type
    ///
    /// ## Safety
    ///
    /// Caller must ensure the two types have the same representation.
    /// **Panics** if sizes don't match (which is not a sufficient check).
    pub(crate) unsafe fn data_subst<B>(self) -> OwnedRepr<B>
    {
        // necessary but not sufficient check
        assert_eq!(mem::size_of::<A>(), mem::size_of::<B>());
        let self_ = ManuallyDrop::new(self);
        OwnedRepr {
            ptr: self_.ptr.cast::<B>(),
            len: self_.len,
            capacity: self_.capacity,
        }
    }

    fn modify_as_vec(&mut self, f: impl FnOnce(Vec<A>) -> Vec<A>)
    {
        let v = self.take_as_vec();
        *self = Self::from(f(v));
    }

    fn take_as_vec(&mut self) -> Vec<A>
    {
        let capacity = self.capacity;
        let len = self.len;
        self.len = 0;
        self.capacity = 0;
        unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), len, capacity) }
    }
}

impl<A> Clone for OwnedRepr<A>
where A: Clone
{
    fn clone(&self) -> Self
    {
        Self::from(self.as_slice().to_owned())
    }

    fn clone_from(&mut self, other: &Self)
    {
        let mut v = self.take_as_vec();
        let other = other.as_slice();

        if v.len() > other.len() {
            v.truncate(other.len());
        }
        let (front, back) = other.split_at(v.len());
        v.clone_from_slice(front);
        v.extend_from_slice(back);
        *self = Self::from(v);
    }
}

impl<A> Drop for OwnedRepr<A>
{
    fn drop(&mut self)
    {
        if self.capacity > 0 {
            // correct because: If the elements don't need dropping, an
            // empty Vec is ok. Only the Vec's allocation needs dropping.
            //
            // implemented because: in some places in ndarray
            // where A: Copy (hence does not need drop) we use uninitialized elements in
            // vectors. Setting the length to 0 avoids that the vector tries to
            // drop, slice or otherwise produce values of these elements.
            // (The details of the validity letting this happen with nonzero len, are
            // under discussion as of this writing.)
            if !mem::needs_drop::<A>() {
                self.len = 0;
            }
            // drop as a Vec.
            self.take_as_vec();
        }
    }
}

unsafe impl<A> Sync for OwnedRepr<A> where A: Sync {}
unsafe impl<A> Send for OwnedRepr<A> where A: Send {}
