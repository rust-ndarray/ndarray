//! Reference type for layouts

use core::ops::{Deref, DerefMut};

use crate::{ArrayBase, LayoutRef, RawData, RefBase};

impl<A, D> AsRef<LayoutRef<A, D>> for LayoutRef<A, D>
{
    fn as_ref(&self) -> &LayoutRef<A, D>
    {
        self
    }
}

impl<A, D> AsMut<LayoutRef<A, D>> for LayoutRef<A, D>
{
    fn as_mut(&mut self) -> &mut LayoutRef<A, D>
    {
        self
    }
}

impl<S, D> AsRef<LayoutRef<S::Elem, D>> for ArrayBase<S, D>
where S: RawData
{
    fn as_ref(&self) -> &LayoutRef<S::Elem, D>
    {
        // SAFETY: The pointer will hold all the guarantees of `as_ref`:
        // - The pointer is aligned because neither type use repr(align)
        // - It is "dereferencable" because it just points to self
        // - For the same reason, it is initialized
        unsafe {
            (&self.layout as *const LayoutRef<S::Elem, D>)
                .cast::<LayoutRef<S::Elem, D>>()
                .as_ref()
        }
        .expect("Pointer to self will always be non-null")
    }
}

impl<S, D> AsMut<LayoutRef<S::Elem, D>> for ArrayBase<S, D>
where S: RawData
{
    fn as_mut(&mut self) -> &mut LayoutRef<S::Elem, D>
    {
        // SAFETY: The pointer will hold all the guarantees of `as_ref`:
        // - The pointer is aligned because neither type use repr(align)
        // - It is "dereferencable" because it just points to self
        // - For the same reason, it is initialized
        unsafe {
            (&mut self.layout as *mut LayoutRef<S::Elem, D>)
                .cast::<LayoutRef<S::Elem, D>>()
                .as_mut()
        }
        .expect("Pointer to self will always be non-null")
    }
}

impl<A, D, R> Deref for RefBase<A, D, R>
{
    type Target = LayoutRef<A, D>;

    fn deref(&self) -> &Self::Target
    {
        unsafe {
            (&self.layout as *const LayoutRef<A, D>)
                .cast::<LayoutRef<A, D>>()
                .as_ref()
        }
        .expect("Pointers to parts will never be null")
    }
}

impl<A, D, R> DerefMut for RefBase<A, D, R>
{
    fn deref_mut(&mut self) -> &mut Self::Target
    {
        unsafe {
            (&mut self.layout as *mut LayoutRef<A, D>)
                .cast::<LayoutRef<A, D>>()
                .as_mut()
        }
        .expect("Pointers to parts will never be null")
    }
}

// Blanket impl for AsRef, so that functions that take
// AsRef<LayoutRef> can take RefBase
impl<T, A, D, R> AsRef<T> for RefBase<A, D, R>
where
    T: ?Sized,
    <RefBase<A, D, R> as Deref>::Target: AsRef<T>,
{
    fn as_ref(&self) -> &T
    {
        self.deref().as_ref()
    }
}

// Blanket impl for AsMut, so that functions that take
// AsMut<LayoutRef> can take RefBase
impl<T, A, D, R> AsMut<T> for RefBase<A, D, R>
where
    T: ?Sized,
    <RefBase<A, D, R> as Deref>::Target: AsMut<T>,
{
    fn as_mut(&mut self) -> &mut T
    {
        self.deref_mut().as_mut()
    }
}

/// # Safety
///
/// Usually the pointer would be bad to just clone, as we'd have aliasing
/// and completely separated references to the same data. However, it is
/// impossible to read the data behind the pointer from a LayoutRef (this
/// is a safety invariant that *must* be maintained), and therefore we can
/// Clone and Copy as desired.
impl<A, D: Clone> Clone for LayoutRef<A, D>
{
    fn clone(&self) -> Self
    {
        Self {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            ptr: self.ptr,
        }
    }
}

impl<A, D: Clone + Copy> Copy for LayoutRef<A, D> {}
