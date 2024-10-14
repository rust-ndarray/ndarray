//! Implementations that connect arrays to their reference types.
//!
//! `ndarray` has four kinds of array types that users may interact with:
//!     1. [`ArrayBase`], which represents arrays which own their layout (shape and strides)
//!     2. [`ArrayRef`], which represents a read-safe, uniquely-owned look at an array
//!     3. [`RawRef`], which represents a read-unsafe, possibly-shared look at an array
//!     4. [`LayoutRef`], which represents a look at an array's underlying structure,
//!         but does not allow data reading of any kind
//!
//! These types are connected through a number of `Deref` and `AsRef` implementations.
//!     1. `ArrayBase<S, D>` dereferences to `ArrayRef` when `S: Data`
//!     2. `ArrayBase<S, D>` mutably dereferences to `ArrayRef` when `S: DataMut`, and ensures uniqueness
//!     3. `ArrayRef` mutably dereferences to `RawRef`
//!     4. `RawRef` mutably dereferences to `LayoutRef`
//! This chain works very well for arrays whose data is safe to read and is uniquely held.
//! Because raw views do not meet `S: Data`, they cannot dereference to `ArrayRef`; furthermore,
//! technical limitations of Rust's compiler means that `ArrayBase` cannot have multiple `Deref` implementations.
//! In addition, shared-data arrays do not want to go down the `Deref` path to get to methods on `RawRef`
//! or `LayoutRef`, since that would unecessarily ensure their uniqueness.
//!
//! To mitigate these problems, `ndarray` also provides `AsRef` and `AsMut` implementations as follows:
//!     1. `ArrayBase` implements `AsRef` to `RawRef` and `LayoutRef` when `S: RawData`
//!     2. `ArrayBase` implements `AsMut` to `RawRef` when `S: RawDataMut`
//!     3. `ArrayBase` implements `AsMut` to `LayoutRef` unconditionally
//!     4. `ArrayRef` implements `AsMut` to `RawRef` and `LayoutRef` unconditionally
//!     5. `RawRef` implements `AsMut` to `LayoutRef`
//!     6. `RawRef` and `LayoutRef` implement `AsMut` to themselves
//!
//! This allows users to write a single method or trait implementation that takes `T: AsRef<RawRef<A, D>>`
//! or `T: AsRef<LayoutRef<A, D>>` and have that functionality work on any of the relevant array types.

use core::ops::{Deref, DerefMut};

use crate::{ArrayBase, ArrayRef, Data, DataMut, Dimension, LayoutRef, RawData, RawDataMut, RawRef};

// D1: &ArrayBase -> &ArrayRef when data is safe to read
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

// D2: &mut ArrayBase -> &mut ArrayRef when data is safe to read; ensure uniqueness
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

// D3: &ArrayRef -> &RawRef
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

// D4: &mut ArrayRef -> &mut RawRef
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

// D5: &RawRef -> &LayoutRef
impl<A, D> Deref for RawRef<A, D>
{
    type Target = LayoutRef<A, D>;

    fn deref(&self) -> &Self::Target
    {
        &self.0
    }
}

// D5: &mut RawRef -> &mut LayoutRef
impl<A, D> DerefMut for RawRef<A, D>
{
    fn deref_mut(&mut self) -> &mut Self::Target
    {
        &mut self.0
    }
}

// A1: &ArrayBase -AR-> &RawRef
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

// A2: &mut ArrayBase -AM-> &mut RawRef
impl<A, S, D> AsMut<RawRef<A, D>> for ArrayBase<S, D>
where S: RawDataMut<Elem = A>
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

// A3: &ArrayBase -AR-> &LayoutRef
impl<A, S, D> AsRef<LayoutRef<A, D>> for ArrayBase<S, D>
where S: RawData<Elem = A>
{
    fn as_ref(&self) -> &LayoutRef<A, D>
    {
        &self.layout
    }
}

// A3: &mut ArrayBase -AM-> &mut LayoutRef
impl<A, S, D> AsMut<LayoutRef<A, D>> for ArrayBase<S, D>
where S: RawData<Elem = A>
{
    fn as_mut(&mut self) -> &mut LayoutRef<A, D>
    {
        &mut self.layout
    }
}

// A4: &ArrayRef -AR-> &RawRef
impl<A, D> AsRef<RawRef<A, D>> for ArrayRef<A, D>
{
    fn as_ref(&self) -> &RawRef<A, D>
    {
        &**self
    }
}

// A4: &mut ArrayRef -AM-> &mut RawRef
impl<A, D> AsMut<RawRef<A, D>> for ArrayRef<A, D>
{
    fn as_mut(&mut self) -> &mut RawRef<A, D>
    {
        &mut **self
    }
}

// A4: &ArrayRef -AR-> &LayoutRef
impl<A, D> AsRef<LayoutRef<A, D>> for ArrayRef<A, D>
{
    fn as_ref(&self) -> &LayoutRef<A, D>
    {
        &***self
    }
}

// A4: &mut ArrayRef -AM-> &mut LayoutRef
impl<A, D> AsMut<LayoutRef<A, D>> for ArrayRef<A, D>
{
    fn as_mut(&mut self) -> &mut LayoutRef<A, D>
    {
        &mut ***self
    }
}

// A5: &RawRef -AR-> &LayoutRef
impl<A, D> AsRef<LayoutRef<A, D>> for RawRef<A, D>
{
    fn as_ref(&self) -> &LayoutRef<A, D>
    {
        &**self
    }
}

// A5: &mut RawRef -AM-> &mut LayoutRef
impl<A, D> AsMut<LayoutRef<A, D>> for RawRef<A, D>
{
    fn as_mut(&mut self) -> &mut LayoutRef<A, D>
    {
        &mut **self
    }
}

// A6: &RawRef -AR-> &RawRef
impl<A, D> AsRef<RawRef<A, D>> for RawRef<A, D>
{
    fn as_ref(&self) -> &RawRef<A, D>
    {
        self
    }
}

// A6: &mut RawRef -AM-> &mut RawRef
impl<A, D> AsMut<RawRef<A, D>> for RawRef<A, D>
{
    fn as_mut(&mut self) -> &mut RawRef<A, D>
    {
        self
    }
}

// A6: &LayoutRef -AR-> &LayoutRef
impl<A, D> AsRef<LayoutRef<A, D>> for LayoutRef<A, D>
{
    fn as_ref(&self) -> &LayoutRef<A, D>
    {
        self
    }
}

// A6: &mut LayoutRef -AR-> &mut LayoutRef
impl<A, D> AsMut<LayoutRef<A, D>> for LayoutRef<A, D>
{
    fn as_mut(&mut self) -> &mut LayoutRef<A, D>
    {
        self
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
