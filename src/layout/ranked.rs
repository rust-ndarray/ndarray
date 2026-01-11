//! Unified trait for type- and runtime-level array rank.
//!
//! This module defines the [`Ranked`] trait, which bridges compile-time and runtime representations
//! of array dimensionality. It enables generic code to query the number of dimensions (rank) of an
//! array, whether known statically (via [`Dimensionality`]) or only at runtime. Blanket
//! implementations are provided for common pointer and container types.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::{
    layout::dimensionality::{Dimensionality, D1},
    ArrayBase,
    ArrayParts,
    ArrayRef,
    LayoutRef,
    RawData,
    RawRef,
};

/// A trait to unify type- and runtime-level number of dimensions.
///
/// The [`Dimensionality`] trait captures array rank at the type level; however it
/// is limited at runtime. If the `Dimensionality` is dynamic (i.e., [`DDyn`][DDyn])
/// then the dimensionality cannot be known at compile time. This trait unifies type-
/// and runtime-level dimensionality by providing:
/// 1. An associated type, [`Rank`][Rank], with type-level dimensionality
/// 2. A function, [`ndim`][ndim], that can give the dimensionality at runtime.
///
/// [DDyn]: crate::layout::dimensionality::DDyn
/// [Rank]: Ranked::Rank
/// [ndim]: Ranked::ndim
/// [N]: Dimensionality::N
pub trait Ranked
{
    /// The compile-time rank of the type; can be [`DDyn`][DDyn] if unknown.
    ///
    /// [DDyn]: crate::layout::dimensionality::DDyn
    type Rank: Dimensionality;

    /// The runtime number of dimensions of the type.
    fn rank(&self) -> usize;
}

mod blanket_impls
{
    use super::*;
    use alloc::rc::Rc;
    use alloc::sync::Arc;

    impl<T> Ranked for &T
    where T: Ranked
    {
        type Rank = T::Rank;

        fn rank(&self) -> usize
        {
            (*self).rank()
        }
    }

    impl<T> Ranked for &mut T
    where T: Ranked
    {
        type Rank = T::Rank;

        fn rank(&self) -> usize
        {
            (**self).rank()
        }
    }

    impl<T> Ranked for Arc<T>
    where T: Ranked
    {
        type Rank = T::Rank;

        fn rank(&self) -> usize
        {
            (**self).rank()
        }
    }

    impl<T> Ranked for Rc<T>
    where T: Ranked
    {
        type Rank = T::Rank;

        fn rank(&self) -> usize
        {
            (**self).rank()
        }
    }

    impl<T> Ranked for Box<T>
    where T: Ranked
    {
        type Rank = T::Rank;

        fn rank(&self) -> usize
        {
            (**self).rank()
        }
    }
}

impl<T> Ranked for [T]
{
    type Rank = D1;

    fn rank(&self) -> usize
    {
        1
    }
}

impl<T> Ranked for Vec<T>
{
    type Rank = D1;

    fn rank(&self) -> usize
    {
        1
    }
}

impl<T, const N: usize> Ranked for [T; N]
{
    type Rank = D1;

    fn rank(&self) -> usize
    {
        1
    }
}

impl<A, D, T: ?Sized> Ranked for ArrayParts<A, D, T>
where D: Ranked
{
    type Rank = D::Rank;

    fn rank(&self) -> usize
    {
        self.dim.rank()
    }
}

impl<S, D> Ranked for ArrayBase<S, D>
where
    S: RawData,
    D: Ranked,
{
    type Rank = D::Rank;

    fn rank(&self) -> usize
    {
        self.parts.rank()
    }
}

impl<A, D> Ranked for LayoutRef<A, D>
where D: Ranked
{
    type Rank = D::Rank;

    fn rank(&self) -> usize
    {
        self.0.rank()
    }
}

impl<A, D> Ranked for ArrayRef<A, D>
where D: Ranked
{
    type Rank = D::Rank;

    fn rank(&self) -> usize
    {
        self.0.rank()
    }
}

impl<A, D> Ranked for RawRef<A, D>
where D: Ranked
{
    type Rank = D::Rank;

    fn rank(&self) -> usize
    {
        self.0.rank()
    }
}
