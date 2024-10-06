//! Code for the array reference type

use core::ops::{Deref, DerefMut};

use crate::{ArrayBase, Dimension, RawData, RawDataMut, RefBase};

/// Unit struct to mark a reference as raw
///
/// Only visible because it is necessary for [`crate::RawRef`]
#[derive(Copy, Clone)]
pub struct Raw;

/// Unit struct to mark a reference as safe
///
/// Only visible because it is necessary for [`crate::ArrRef`]
#[derive(Copy, Clone)]
pub struct Safe;

/// A trait for array references that adhere to the basic constraints of `ndarray`.
///
/// Cannot be implemented outside of `ndarray`.
pub trait RawReferent
{
    private_decl! {}
}

/// A trait for array references that point to data that is safe to read.
///
/// Cannot be implemented outside of `ndarray`.
pub trait Referent
{
    private_decl! {}
}

impl RawReferent for Raw
{
    private_impl! {}
}
impl RawReferent for Safe
{
    private_impl! {}
}
impl Referent for Safe
{
    private_impl! {}
}

impl<S, D> Deref for ArrayBase<S, D>
where S: RawData
{
    type Target = RefBase<S::Elem, D, S::Referent>;

    fn deref(&self) -> &Self::Target
    {
        // SAFETY: The pointer will hold all the guarantees of `as_ref`:
        // - The pointer is aligned because neither type use repr(align)
        // - It is "dereferencable" because it just points to self
        // - For the same reason, it is initialized
        unsafe {
            (self as *const Self)
                .cast::<RefBase<S::Elem, D, S::Referent>>()
                .as_ref()
        }
        .expect("Pointer to self will always be non-null")
    }
}

impl<S, D> DerefMut for ArrayBase<S, D>
where
    S: RawDataMut,
    D: Dimension,
{
    fn deref_mut(&mut self) -> &mut Self::Target
    {
        self.try_ensure_unique();
        // SAFETY: The pointer will hold all the guarantees of `as_ref`:
        // - The pointer is aligned because neither type use repr(align)
        // - It is "dereferencable" because it just points to self
        // - For the same reason, it is initialized
        unsafe {
            (self as *mut Self)
                .cast::<RefBase<S::Elem, D, S::Referent>>()
                .as_mut()
        }
        .expect("Pointer to self will always be non-null")
    }
}
