//! Code for the array reference type

use core::ops::{Deref, DerefMut};

use crate::{ArrayBase, Dimension, RawData, RawDataMut, RefBase};

/// Unit struct to mark a reference as raw
#[derive(Copy, Clone)]
pub struct Raw;
/// Unit struct to mark a reference as safe
#[derive(Copy, Clone)]
pub struct Safe;

pub trait RawReferent {
    private_decl! {}
}
pub trait Referent {
    private_decl! {}
}

impl RawReferent for Raw {
    private_impl! {}
}
impl RawReferent for Safe {
    private_impl! {}
}
impl Referent for Safe {
    private_impl! {}
}

impl<S, D> Deref for ArrayBase<S, D>
where S: RawData
{
    type Target = RefBase<S::Elem, D, S::Referent>;

    fn deref(&self) -> &Self::Target {
        &self.aref
    }
}

impl<S, D> DerefMut for ArrayBase<S, D>
where S: RawDataMut, D: Dimension,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.try_ensure_unique();
        &mut self.aref
    }
}
