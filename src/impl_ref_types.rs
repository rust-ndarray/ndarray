//! Code for the array reference type

use core::ops::{Deref, DerefMut};

use crate::{ArrayBase, ArrayRef, Data, DataMut, Dimension, LayoutRef, RawData, RawRef};

impl<S, D> Deref for ArrayBase<S, D>
where S: Data
{
    type Target = ArrayRef<S::Elem, D>;

    fn deref(&self) -> &Self::Target
    {
        // SAFETY: The pointer will hold all the guarantees of `as_ref`:
        // - The pointer is aligned because neither type use repr(align)
        // - It is "dereferencable" because it just points to self
        // - For the same reason, it is initialized
        unsafe {
            (&self.layout as *const LayoutRef<S::Elem, D>)
                .cast::<ArrayRef<S::Elem, D>>()
                .as_ref()
        }
        .expect("References are always non-null")
    }
}

impl<S, D> DerefMut for ArrayBase<S, D>
where
    S: DataMut,
    D: Dimension,
{
    fn deref_mut(&mut self) -> &mut Self::Target
    {
        self.ensure_unique();
        // SAFETY: The pointer will hold all the guarantees of `as_ref`:
        // - The pointer is aligned because neither type use repr(align)
        // - It is "dereferencable" because it just points to self
        // - For the same reason, it is initialized
        unsafe {
            (&mut self.layout as *mut LayoutRef<S::Elem, D>)
                .cast::<ArrayRef<S::Elem, D>>()
                .as_mut()
        }
        .expect("References are always non-null")
    }
}

impl<A, S, D> AsRef<RawRef<A, D>> for ArrayBase<S, D>
where S: RawData<Elem = A>
{
    fn as_ref(&self) -> &RawRef<A, D>
    {
        unsafe {
            (&self.layout as *const LayoutRef<A, D>)
                .cast::<RawRef<A, D>>()
                .as_ref()
        }
        .expect("References are always non-null")
    }
}

impl<A, S, D> AsMut<RawRef<A, D>> for ArrayBase<S, D>
where S: RawData<Elem = A>
{
    fn as_mut(&mut self) -> &mut RawRef<A, D>
    {
        unsafe {
            (&mut self.layout as *mut LayoutRef<A, D>)
                .cast::<RawRef<A, D>>()
                .as_mut()
        }
        .expect("References are always non-null")
    }
}

impl<A, D> AsRef<RawRef<A, D>> for RawRef<A, D>
{
    fn as_ref(&self) -> &RawRef<A, D>
    {
        self
    }
}

impl<A, D> AsMut<RawRef<A, D>> for RawRef<A, D>
{
    fn as_mut(&mut self) -> &mut RawRef<A, D>
    {
        self
    }
}

impl<A, D> Deref for ArrayRef<A, D>
{
    type Target = RawRef<A, D>;

    fn deref(&self) -> &Self::Target
    {
        unsafe {
            (self as *const ArrayRef<A, D>)
                .cast::<RawRef<A, D>>()
                .as_ref()
        }
        .expect("References are always non-null")
    }
}

impl<A, D> DerefMut for ArrayRef<A, D>
{
    fn deref_mut(&mut self) -> &mut Self::Target
    {
        unsafe {
            (self as *mut ArrayRef<A, D>)
                .cast::<RawRef<A, D>>()
                .as_mut()
        }
        .expect("References are always non-null")
    }
}

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

impl<A, D> Deref for RawRef<A, D>
{
    type Target = LayoutRef<A, D>;

    fn deref(&self) -> &Self::Target
    {
        &self.0
    }
}

impl<A, D> DerefMut for RawRef<A, D>
{
    fn deref_mut(&mut self) -> &mut Self::Target
    {
        &mut self.0
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
