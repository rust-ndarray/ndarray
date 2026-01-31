//! Unified trait for type- and runtime-level array rank.
//!
//! This module defines the [`Ranked`] trait, which bridges compile-time and runtime representations
//! of array dimensionality. It enables generic code to query the number of dimensions (rank) of an
//! array, whether known statically (via [`Rank`]) or only at runtime. Blanket
//! implementations are provided for common pointer and container types.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::{
    layout::rank::{Rank, R1},
    ArrayBase,
    ArrayParts,
    ArrayRef,
    LayoutRef,
    RawData,
    RawRef,
};

/// A trait to unify type- and runtime-level number of dimensions.
///
/// The [`Rank`] trait captures array rank at the type level; however it
/// is limited at runtime. If the `Rank` is dynamic (i.e., [`DynRank`][DynRank])
/// then the dimensionality cannot be known at compile time. This trait unifies type-
/// and runtime-level dimensionality by providing:
/// 1. An associated type, [`NDim`][NDim], with type-level dimensionality
/// 2. A function, [`ndim`][ndim], that can give the dimensionality at runtime.
///
/// [DynRank]: crate::layout::rank::DynRank
/// [NDim]: Ranked::NDim
/// [ndim]: Ranked::ndim
pub trait Ranked
{
    /// The compile-time rank of the type; can be [`DynRank`][DynRank] if unknown.
    ///
    /// [DynRank]: crate::layout::dimensionality::DynRank
    type NDim: Rank;

    /// The runtime number of dimensions of the type.
    fn ndim(&self) -> usize;
}

mod blanket_impls
{
    use super::*;
    use alloc::rc::Rc;

    #[cfg(target_has_atomic = "ptr")]
    use alloc::sync::Arc;
    #[cfg(not(target_has_atomic = "ptr"))]
    use portable_atomic_util::Arc;

    impl<T> Ranked for &T
    where T: Ranked
    {
        type NDim = T::NDim;

        fn ndim(&self) -> usize
        {
            (*self).ndim()
        }
    }

    impl<T> Ranked for &mut T
    where T: Ranked
    {
        type NDim = T::NDim;

        fn ndim(&self) -> usize
        {
            (**self).ndim()
        }
    }

    impl<T> Ranked for Arc<T>
    where T: Ranked
    {
        type NDim = T::NDim;

        fn ndim(&self) -> usize
        {
            (**self).ndim()
        }
    }

    impl<T> Ranked for Rc<T>
    where T: Ranked
    {
        type NDim = T::NDim;

        fn ndim(&self) -> usize
        {
            (**self).ndim()
        }
    }

    impl<T> Ranked for Box<T>
    where T: Ranked
    {
        type NDim = T::NDim;

        fn ndim(&self) -> usize
        {
            (**self).ndim()
        }
    }
}

impl<T> Ranked for [T]
{
    type NDim = R1;

    fn ndim(&self) -> usize
    {
        1
    }
}

impl<T> Ranked for Vec<T>
{
    type NDim = R1;

    fn ndim(&self) -> usize
    {
        1
    }
}

impl<T, const N: usize> Ranked for [T; N]
{
    type NDim = R1;

    fn ndim(&self) -> usize
    {
        1
    }
}

impl<A, D, T: ?Sized> Ranked for ArrayParts<A, D, T>
where D: Ranked
{
    type NDim = D::NDim;

    fn ndim(&self) -> usize
    {
        self.dim.ndim()
    }
}

impl<S, D> Ranked for ArrayBase<S, D>
where
    S: RawData,
    D: Ranked,
{
    type NDim = D::NDim;

    fn ndim(&self) -> usize
    {
        self.parts.ndim()
    }
}

impl<A, D> Ranked for LayoutRef<A, D>
where D: Ranked
{
    type NDim = D::NDim;

    fn ndim(&self) -> usize
    {
        self.0.ndim()
    }
}

impl<A, D> Ranked for ArrayRef<A, D>
where D: Ranked
{
    type NDim = D::NDim;

    fn ndim(&self) -> usize
    {
        self.0.ndim()
    }
}

impl<A, D> Ranked for RawRef<A, D>
where D: Ranked
{
    type NDim = D::NDim;

    fn ndim(&self) -> usize
    {
        self.0.ndim()
    }
}
