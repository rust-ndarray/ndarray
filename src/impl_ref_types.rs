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
//!     3. `ArrayBase` implements `AsRef` and `AsMut` to `LayoutRef` unconditionally
//!     4. `ArrayRef` implements `AsRef` and `AsMut` to `RawRef` and `LayoutRef` unconditionally
//!     5. `RawRef` implements `AsRef` and `AsMut` to `LayoutRef`
//!     6. `RawRef` and `LayoutRef` implement `AsRef` and `AsMut` to themselves
//!
//! This allows users to write a single method or trait implementation that takes `T: AsRef<RawRef<A, D>>`
//! or `T: AsRef<LayoutRef<A, D>>` and have that functionality work on any of the relevant array types.

use alloc::borrow::ToOwned;
use core::{
    borrow::{Borrow, BorrowMut},
    ops::{Deref, DerefMut},
};

use crate::{
    Array,
    ArrayBase,
    ArrayPartsSized,
    ArrayPartsUnsized,
    ArrayRef,
    Data,
    DataMut,
    Dimension,
    LayoutRef,
    RawData,
    RawDataMut,
    RawRef,
};

// D1: &ArrayBase -> &ArrayRef when data is safe to read
impl<S, D> Deref for ArrayBase<S, D>
where S: Data
{
    type Target = ArrayRef<S::Elem, D>;

    fn deref(&self) -> &Self::Target
    {
        // SAFETY:
        // - The pointer is aligned because neither type uses repr(align)
        // - It is "dereferencable" because it comes from a reference
        // - For the same reason, it is initialized
        // - The cast is valid because ArrayRef uses #[repr(transparent)]
        let parts: &ArrayPartsUnsized<S::Elem, D> = &self.parts;
        let ptr = (parts as *const ArrayPartsUnsized<S::Elem, D>) as *const ArrayRef<S::Elem, D>;
        unsafe { &*ptr }
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
        // SAFETY:
        // - The pointer is aligned because neither type uses repr(align)
        // - It is "dereferencable" because it comes from a reference
        // - For the same reason, it is initialized
        // - The cast is valid because ArrayRef uses #[repr(transparent)]
        let parts: &mut ArrayPartsUnsized<S::Elem, D> = &mut self.parts;
        let ptr = (parts as *mut ArrayPartsUnsized<S::Elem, D>) as *mut ArrayRef<S::Elem, D>;
        unsafe { &mut *ptr }
    }
}

// D3: &ArrayRef -> &RawRef
impl<A, D> Deref for ArrayRef<A, D>
{
    type Target = RawRef<A, D>;

    fn deref(&self) -> &Self::Target
    {
        // SAFETY:
        // - The pointer is aligned because neither type uses repr(align)
        // - It is "dereferencable" because it comes from a reference
        // - For the same reason, it is initialized
        // - The cast is valid because ArrayRef uses #[repr(transparent)]
        unsafe { &*((self as *const ArrayRef<A, D>) as *const RawRef<A, D>) }
    }
}

// D4: &mut ArrayRef -> &mut RawRef
impl<A, D> DerefMut for ArrayRef<A, D>
{
    fn deref_mut(&mut self) -> &mut Self::Target
    {
        // SAFETY:
        // - The pointer is aligned because neither type uses repr(align)
        // - It is "dereferencable" because it comes from a reference
        // - For the same reason, it is initialized
        // - The cast is valid because ArrayRef uses #[repr(transparent)]
        unsafe { &mut *((self as *mut ArrayRef<A, D>) as *mut RawRef<A, D>) }
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
        // SAFETY:
        // - The pointer is aligned because neither type uses repr(align)
        // - It is "dereferencable" because it comes from a reference
        // - For the same reason, it is initialized
        // - The cast is valid because ArrayRef uses #[repr(transparent)]
        let parts: &ArrayPartsUnsized<S::Elem, D> = &self.parts;
        let ptr = (parts as *const ArrayPartsUnsized<S::Elem, D>) as *const RawRef<S::Elem, D>;
        unsafe { &*ptr }
    }
}

// A2: &mut ArrayBase -AM-> &mut RawRef
impl<A, S, D> AsMut<RawRef<A, D>> for ArrayBase<S, D>
where S: RawDataMut<Elem = A>
{
    fn as_mut(&mut self) -> &mut RawRef<A, D>
    {
        // SAFETY:
        // - The pointer is aligned because neither type uses repr(align)
        // - It is "dereferencable" because it comes from a reference
        // - For the same reason, it is initialized
        // - The cast is valid because ArrayRef uses #[repr(transparent)]
        let parts: &mut ArrayPartsUnsized<S::Elem, D> = &mut self.parts;
        let ptr = (parts as *mut ArrayPartsUnsized<S::Elem, D>) as *mut RawRef<S::Elem, D>;
        unsafe { &mut *ptr }
    }
}

// A3: &ArrayBase -AR-> &LayoutRef
impl<A, S, D> AsRef<LayoutRef<A, D>> for ArrayBase<S, D>
where S: RawData<Elem = A>
{
    fn as_ref(&self) -> &LayoutRef<A, D>
    {
        let parts: &ArrayPartsUnsized<S::Elem, D> = &self.parts;
        let ptr = (parts as *const ArrayPartsUnsized<S::Elem, D>) as *const LayoutRef<S::Elem, D>;
        unsafe { &*ptr }
    }
}

// A3: &mut ArrayBase -AM-> &mut LayoutRef
impl<A, S, D> AsMut<LayoutRef<A, D>> for ArrayBase<S, D>
where S: RawData<Elem = A>
{
    fn as_mut(&mut self) -> &mut LayoutRef<A, D>
    {
        let parts: &mut ArrayPartsUnsized<S::Elem, D> = &mut self.parts;
        let ptr = (parts as *mut ArrayPartsUnsized<S::Elem, D>) as *mut LayoutRef<S::Elem, D>;
        unsafe { &mut *ptr }
    }
}

// A4: &ArrayRef -AR-> &RawRef
impl<A, D> AsRef<RawRef<A, D>> for ArrayRef<A, D>
{
    fn as_ref(&self) -> &RawRef<A, D>
    {
        self
    }
}

// A4: &mut ArrayRef -AM-> &mut RawRef
impl<A, D> AsMut<RawRef<A, D>> for ArrayRef<A, D>
{
    fn as_mut(&mut self) -> &mut RawRef<A, D>
    {
        self
    }
}

// A4: &ArrayRef -AR-> &LayoutRef
impl<A, D> AsRef<LayoutRef<A, D>> for ArrayRef<A, D>
{
    fn as_ref(&self) -> &LayoutRef<A, D>
    {
        self
    }
}

// A4: &mut ArrayRef -AM-> &mut LayoutRef
impl<A, D> AsMut<LayoutRef<A, D>> for ArrayRef<A, D>
{
    fn as_mut(&mut self) -> &mut LayoutRef<A, D>
    {
        self
    }
}

// A5: &RawRef -AR-> &LayoutRef
impl<A, D> AsRef<LayoutRef<A, D>> for RawRef<A, D>
{
    fn as_ref(&self) -> &LayoutRef<A, D>
    {
        self
    }
}

// A5: &mut RawRef -AM-> &mut LayoutRef
impl<A, D> AsMut<LayoutRef<A, D>> for RawRef<A, D>
{
    fn as_mut(&mut self) -> &mut LayoutRef<A, D>
    {
        self
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
impl<A, D: Clone> Clone for ArrayPartsSized<A, D>
{
    fn clone(&self) -> Self
    {
        Self {
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            ptr: self.ptr,
            _dst_control: [0; 0],
        }
    }
}

impl<A, D: Clone + Copy> Copy for ArrayPartsSized<A, D> {}

impl<S, D> Borrow<RawRef<S::Elem, D>> for ArrayBase<S, D>
where S: RawData
{
    fn borrow(&self) -> &RawRef<S::Elem, D>
    {
        self.as_ref()
    }
}

impl<S, D> BorrowMut<RawRef<S::Elem, D>> for ArrayBase<S, D>
where S: RawDataMut
{
    fn borrow_mut(&mut self) -> &mut RawRef<S::Elem, D>
    {
        self.as_mut()
    }
}

impl<S, D> Borrow<ArrayRef<S::Elem, D>> for ArrayBase<S, D>
where S: Data
{
    fn borrow(&self) -> &ArrayRef<S::Elem, D>
    {
        self
    }
}

impl<S, D> BorrowMut<ArrayRef<S::Elem, D>> for ArrayBase<S, D>
where
    S: DataMut,
    D: Dimension,
{
    fn borrow_mut(&mut self) -> &mut ArrayRef<S::Elem, D>
    {
        self
    }
}

impl<A, D> ToOwned for ArrayRef<A, D>
where
    A: Clone,
    D: Dimension,
{
    type Owned = Array<A, D>;

    fn to_owned(&self) -> Self::Owned
    {
        self.to_owned()
    }

    fn clone_into(&self, target: &mut Array<A, D>)
    {
        target.zip_mut_with(self, |tgt, src| tgt.clone_from(src));
    }
}

/// Shortcuts for the various as_ref calls
impl<A, S, D> ArrayBase<S, D>
where S: RawData<Elem = A>
{
    /// Cheaply convert a reference to the array to an &LayoutRef
    pub fn as_layout_ref(&self) -> &LayoutRef<A, D>
    {
        self.as_ref()
    }

    /// Cheaply and mutably convert a reference to the array to an &LayoutRef
    pub fn as_layout_ref_mut(&mut self) -> &mut LayoutRef<A, D>
    {
        self.as_mut()
    }

    /// Cheaply convert a reference to the array to an &RawRef
    pub fn as_raw_ref(&self) -> &RawRef<A, D>
    {
        self.as_ref()
    }

    /// Cheaply and mutably convert a reference to the array to an &RawRef
    pub fn as_raw_ref_mut(&mut self) -> &mut RawRef<A, D>
    where S: RawDataMut<Elem = A>
    {
        self.as_mut()
    }
}

macro_rules! blanket_ref_alias {
    ($alias_name:ident, $alias_name_mut:ident, $target:ty) => {
        pub trait $alias_name<A, D>: AsRef<$target> {}

        impl<T: ?Sized, A, D> $alias_name<A, D> for T where T: AsRef<$target> {}

        pub trait $alias_name_mut<A, D>: AsMut<$target> {}

        impl<T: ?Sized, A, D> $alias_name_mut<A, D> for T where T: AsMut<$target> {}
    };
}

blanket_ref_alias!(AsLayoutRef, AsMutLayoutRef, LayoutRef<A, D>);
blanket_ref_alias!(AsRawRef, AsMutRawRef, LayoutRef<A, D>);

/// Tests that a mem::swap can't compile by putting it into a doctest
///
/// ```compile_fail
/// let mut x = Array1::from_vec(vec![0, 1, 2]);
/// {
///     let mut y = Array1::from_vec(vec![4, 5, 6]);
///     let x_ref = x.as_layout_ref_mut();
///     let y_ref = y.as_layout_ref_mut();
///     core::mem::swap(x_ref, y_ref);
/// }
/// ```
#[allow(dead_code)]
fn test_no_swap_via_doctests() {}
