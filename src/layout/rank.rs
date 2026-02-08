//! Type-level representations of array dimensionality.
//!
//! This module defines the [`Rank`] trait and related types used to represent
//! the number of axes an array has, either at compile time ([`ConstRank`]) or dynamically
//! ([`DynRank`]). These types support basic type-level operations such as addition and
//! maximum, which are used to model array operations like concatenation and broadcasting.

use core::fmt::Debug;

/// A trait representing a dimensionality, i.e., an unsigned integer indicating how many axes an array has.
///
/// `ndarray` encodes an array’s dimensionality in the type system when possible, which is useful for
/// debugging and for writing generic array code. However, some operations produce arrays whose
/// dimensionality cannot be known at compile time. This trait provides a common abstraction for both
/// statically known and dynamic dimensionalities.
///
/// Compile-time dimensionalities are currently supported for values from 0 to 12, inclusive.
/// Any dimensionality above 12 must be represented with [`DynRank`], even if it is known at compile time.
///
/// The `Smaller` and `Larger` associated types allow users to move to adjacent dimensionalities at the type level.
///
/// ## Dynamic dimensionalities
/// A type implementing `Rank` does not expose its dimensionality as a runtime value.
/// In dynamic cases, `DynRank` means that the dimensionality is not known at compile time.
/// The actual number of axes is taken directly from the array’s shape.
pub trait Rank:
    Copy
    + Eq
    + Debug
    + Send
    + Sync
    + RMax<R0, Output = Self>
    + RMax<Self, Output = Self>
    + RMax<DynRank, Output = DynRank>
    + RMax<Self::Smaller, Output = Self>
    + RMax<Self::Larger, Output = Self::Larger>
    + RAdd<Self>
    + RAdd<Self::Smaller>
    + RAdd<Self::Larger>
    + RAdd<R0, Output = Self>
    + RAdd<R1, Output = Self::Larger>
    + RAdd<DynRank, Output = DynRank>
{
    /// The dimensionality as a constant `usize`, or `None` if it is dynamic.
    const N: Option<usize>;

    /// The next-smaller possible dimensionality.
    ///
    /// For the smallest possible dimensionality (currently 0-dimensional), there
    /// is of course no "smaller" dimensionality. Instead, `ConstRank::<0>::Smaller` just
    /// refers back to `ConstRank<0>`; in other words, it uses a "base case" of 0-dimensionality.
    type Smaller: Rank;

    /// The next-larger dimensionality.
    ///
    /// For the largest compile-time dimensionality (currently 12-dimensional), there
    /// is no "larger" compile-time dimensionality. Instead, `ConstRank::<12>::Larger` just
    /// refers to `DynRank`; in other words, it "escapes" to a dynamically-determined dimensionality.
    type Larger: Rank;
}

/// Adds two dimensionalities at compile time.
///
/// The addition of a constant dimensionality with a dynamic dimensionality
/// will always result in a dynamic dimensionality, effectively "erasing"
/// the compile-time knowledge.
///
/// This type is analogous to the existing [`crate::DimAdd`], but specifically
/// for dimensionality instead of `Dimension` types.
///
/// ## Example
/// ```
/// use ndarray::layout::rank::*;
/// use core::any::TypeId;
///
/// type Added = <R1 as RAdd<R2>>::Output;
/// assert_eq!(TypeId::of::<Added>(), TypeId::of::<R3>());
///
/// type AddedDyn = <R1 as RAdd<DynRank>>::Output;
/// assert_eq!(TypeId::of::<AddedDyn>(), TypeId::of::<DynRank>());
/// ```
pub trait RAdd<D>
{
    /// The result of the type-level addition of two dimensionalities.
    type Output: Rank;
}

/// Takes the maximum of two dimensionalities at compile time.
///
/// The maximum of a constant dimensionality and a dynamic dimensionality
/// will always result in a dynamic dimensionality, effectively "erasing"
/// the compile-time knowledge.
///
/// This type is analogous to the existing [`crate::DimMax`], but specifically
/// for dimensionality instead of `Dimension` types.
///
/// ## Example
/// ```
/// use ndarray::layout::rank::*;
/// use core::any::TypeId;
///
/// type Added = <R1 as RMax<R2>>::Output;
/// assert_eq!(TypeId::of::<Added>(), TypeId::of::<R2>());
///
/// type AddedDyn = <R1 as RMax<DynRank>>::Output;
/// assert_eq!(TypeId::of::<AddedDyn>(), TypeId::of::<DynRank>());
/// ```
pub trait RMax<D>
{
    /// The result of the type-level maximum of two dimensionalities.
    type Output: Rank;
}

/// The N-dimensional static dimensionality.
///
/// This type captures dimensionalities that are known at compile-time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConstRank<const N: usize>;

/// The 0-dimensionality, for "dimensionless" arrays with a single value.
///
/// See [`Rank`] and [`ConstRank`] for more information.
pub type R0 = ConstRank<0>;

macro_rules! def_d_aliases {
    ($(($alias:ident, $N:literal)),*) => {
        $(
            /// A dimensionality for arrays that are
            #[doc = stringify!($N)]
            /// D.
            /// 
            /// See [`Rank`] and [`ConstRank`] for more information.
            pub type $alias = ConstRank<$N>;
        )+
    };
}

def_d_aliases!(
    (R1, 1),
    (R2, 2),
    (R3, 3),
    (R4, 4),
    (R5, 5),
    (R6, 6),
    (R7, 7),
    (R8, 8),
    (R9, 9),
    (R10, 10),
    (R11, 11),
    (R12, 12)
);

/// Implement addition for a given dimensionality.
macro_rules! impl_add {
    ($left:literal, ($($right:literal),*), ddyn: ($($rightd:literal),*)) => {
        // $left + $right still gets you a compile-time dimension
        $(
            impl RAdd<ConstRank<$right>> for ConstRank<$left>
            {
                type Output = ConstRank<{$left + $right}>;
            }
        )*

        // $left + $rightd gets you a dynamic dimensionality
        $(
            impl RAdd<ConstRank<$rightd>> for ConstRank<$left>
            {
                type Output = DynRank;
            }
        )*
    };
}

// There's got to be a macro way to do this in one line to help with
// any future additions of extra dimenions, although it might
// also slow down compile times.
impl_add!(0, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), ddyn: ());
impl_add!(1, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), ddyn: (12));
impl_add!(2, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), ddyn: (11, 12));
impl_add!(3, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), ddyn: (10, 11, 12));
impl_add!(4, (0, 1, 2, 3, 4, 5, 6, 7, 8), ddyn: (9, 10, 11, 12));
impl_add!(5, (0, 1, 2, 3, 4, 5, 6, 7), ddyn: (8, 9, 10, 11, 12));
impl_add!(6, (0, 1, 2, 3, 4, 5, 6), ddyn: (7, 8, 9, 10, 11, 12));
impl_add!(7, (0, 1, 2, 3, 4, 5), ddyn: (6, 7, 8, 9, 10, 11, 12));
impl_add!(8, (0, 1, 2, 3, 4), ddyn: (5, 6, 7, 8, 9, 10, 11, 12));
impl_add!(9, (0, 1, 2, 3), ddyn: (4, 5, 6, 7, 8, 9, 10, 11, 12));
impl_add!(10, (0, 1, 2), ddyn: (3, 4, 5, 6, 7, 8, 9, 10, 11, 12));
impl_add!(11, (0, 1), ddyn: (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));
impl_add!(12, (0), ddyn: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));

macro_rules! impl_max {
    // Base case, just a target with some lowers
    ($($lower:literal),+, target: $target:literal) => {
        $(
            impl RMax<ConstRank<$lower>> for ConstRank<$target>
            {
                type Output = ConstRank<$target>;
            }
        )+
    };
    // General case: at least one lower, at least one upper
    ($($lower:literal),+$(,)? target: $target:literal, $first_upper:literal$(, $($upper:literal),+)?) => {
        $(
            impl RMax<ConstRank<$lower>> for ConstRank<$target>
            {
                type Output = ConstRank<$target>;
            }
        )+
        impl RMax<ConstRank<$first_upper>> for ConstRank<$target>
        {
            type Output = ConstRank<$first_upper>;
        }
        $(
            $(
                impl RMax<ConstRank<$upper>> for ConstRank<$target>
                {
                    type Output = ConstRank<$upper>;
                }
            )+
        )?
        impl_max!($($lower),+, $target, target: $first_upper$(, $($upper),+)?);
    };
    // Helper syntax: zero lowers, target, at least one upper
    (target: $target:literal, $first_upper:literal, $($upper:literal),+) => {
        impl RMax<ConstRank<$first_upper>> for ConstRank<$target>
        {
            type Output = ConstRank<$first_upper>;
        }
        $(
            impl RMax<ConstRank<$upper>> for ConstRank<$target>
            {
                type Output = ConstRank<$upper>;
            }
        )+
        impl_max!($target, target: $first_upper, $($upper),+);
    };
}

impl_max!(target: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

impl<const N: usize> RMax<ConstRank<N>> for ConstRank<N>
where ConstRank<N>: Rank
{
    type Output = Self;
}

macro_rules! impl_dimensionality {
    ($($d:literal),+) => {
        $(
            impl Rank for ConstRank<$d>
            {
                const N: Option<usize> = Some($d);

                type Smaller = ConstRank<{$d - 1}>;

                type Larger = ConstRank<{$d + 1}>;
            }
        )+
    };
}

impl Rank for R0
{
    const N: Option<usize> = Some(0);

    type Smaller = Self;

    type Larger = R1;
}

impl_dimensionality!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

impl Rank for ConstRank<12>
{
    const N: Option<usize> = Some(12);

    type Smaller = R11;

    type Larger = DynRank;
}

/// The dynamic dimensionality.
///
/// This type captures dimensionalities that are unknown at compile-time.
/// See [`Rank`] for more information.
///
/// This type does not carry any information about runtime dimensionality,
/// it just indicate that dimensionality is not known at compile-time.
/// This is done to avoid multiple sources of truth for runtime array
/// dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DynRank;

impl Rank for DynRank
{
    const N: Option<usize> = None;

    type Smaller = Self;

    type Larger = Self;
}

impl RAdd<DynRank> for DynRank
{
    type Output = DynRank;
}

impl<const N: usize> RAdd<ConstRank<N>> for DynRank
{
    type Output = DynRank;
}

impl<const N: usize> RAdd<DynRank> for ConstRank<N>
{
    type Output = DynRank;
}

impl RMax<DynRank> for DynRank
{
    type Output = DynRank;
}

impl<const N: usize> RMax<ConstRank<N>> for DynRank
{
    type Output = DynRank;
}

impl<const N: usize> RMax<DynRank> for ConstRank<N>
{
    type Output = DynRank;
}
