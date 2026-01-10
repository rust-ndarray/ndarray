//! Type-level representations of array dimensionality.
//!
//! This module defines the [`Dimensionality`] trait and related types used to represent
//! the number of axes an array has, either at compile time ([`NDim`]) or dynamically
//! ([`DDyn`]). These types support basic type-level operations such as addition and
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
/// Any dimensionality above 12 must be represented with [`DDyn`], even if it is known at compile time.
///
/// The `Smaller` and `Larger` associated types allow users to move to adjacent dimensionalities at the type level.
///
/// ## Dynamic dimensionalities
/// A type implementing `Dimensionality` does not expose its dimensionality as a runtime value.
/// In dynamic cases, `DDyn` means that the dimensionality is not known at compile time.
/// The actual number of axes is taken directly from the array’s shape.
pub trait Dimensionality:
    Copy
    + Eq
    + Debug
    + Send
    + Sync
    + DMax<D0, Output = Self>
    + DMax<Self, Output = Self>
    + DMax<DDyn, Output = DDyn>
    + DMax<Self::Smaller, Output = Self>
    + DMax<Self::Larger, Output = Self::Larger>
    + DAdd<Self>
    + DAdd<Self::Smaller>
    + DAdd<Self::Larger>
    + DAdd<D0, Output = Self>
    + DAdd<D1, Output = Self::Larger>
    + DAdd<DDyn, Output = DDyn>
{
    /// The dimensionality as a constant `usize`, or `None` if it is dynamic.
    const N: Option<usize>;

    /// The next-smaller possible dimensionality.
    ///
    /// For the smallest possible dimensionality (currently 0-dimensional), there
    /// is of course no "smaller" dimensionality. Instead, `NDim::<0>::Smaller` just
    /// refers back to `NDim<0>`; in other words, it uses a "base case" of 0-dimensionality.
    type Smaller: Dimensionality;

    /// The next-larger dimensionality.
    ///
    /// For the largest compile-time dimensionality (currently 12-dimensional), there
    /// is no "larger" compile-time dimensionality. Instead, `NDim::<12>::Larger` just
    /// refers to `DDyn`; in other words, it "escapes" to a dynamically-determined dimensionality.
    type Larger: Dimensionality;
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
/// use ndarray::layout::dimensionality::*;
/// use core::any::TypeId;
///
/// type Added = <D1 as DAdd<D2>>::Output;
/// assert_eq!(TypeId::of::<Added>(), TypeId::of::<D3>());
///
/// type AddedDyn = <D1 as DAdd<DDyn>>::Output;
/// assert_eq!(TypeId::of::<AddedDyn>(), TypeId::of::<DDyn>());
/// ```
pub trait DAdd<D>
{
    /// The result of the type-level addition of two dimensionalities.
    type Output: Dimensionality;
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
/// use ndarray::layout::dimensionality::*;
/// use core::any::TypeId;
///
/// type Added = <D1 as DMax<D2>>::Output;
/// assert_eq!(TypeId::of::<Added>(), TypeId::of::<D2>());
///
/// type AddedDyn = <D1 as DMax<DDyn>>::Output;
/// assert_eq!(TypeId::of::<AddedDyn>(), TypeId::of::<DDyn>());
/// ```
pub trait DMax<D>
{
    /// The result of the type-level maximum of two dimensionalities.
    type Output: Dimensionality;
}

/// The N-dimensional static dimensionality.
///
/// This type captures dimensionalities that are known at compile-time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NDim<const N: usize>;

/// The 0-dimensionality, for "dimensionless" arrays with a single value.
///
/// See [`Dimensionality`] and [`NDim`] for more information.
pub type D0 = NDim<0>;

macro_rules! def_d_aliases {
    ($(($alias:ident, $N:literal)),*) => {
        $(
            /// A dimensionality for arrays that are
            #[doc = stringify!($N)]
            /// D.
            /// 
            /// See [`Dimensionality`] and [`NDim`] for more information.
            pub type $alias = NDim<$N>;
        )+
    };
}

def_d_aliases!(
    (D1, 1),
    (D2, 2),
    (D3, 3),
    (D4, 4),
    (D5, 5),
    (D6, 6),
    (D7, 7),
    (D8, 8),
    (D9, 9),
    (D10, 10),
    (D11, 11),
    (D12, 12)
);

/// Implement addition for a given dimensionality.
macro_rules! impl_add {
    ($left:literal, ($($right:literal),*), ddyn: ($($rightd:literal),*)) => {
        // $left + $right still gets you a compile-time dimension
        $(
            impl DAdd<NDim<$right>> for NDim<$left>
            {
                type Output = NDim<{$left + $right}>;
            }
        )*

        // $left + $rightd gets you a dynamic dimensionality
        $(
            impl DAdd<NDim<$rightd>> for NDim<$left>
            {
                type Output = DDyn;
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
            impl DMax<NDim<$lower>> for NDim<$target>
            {
                type Output = NDim<$target>;
            }
        )+
    };
    // General case: at least one lower, at least one upper
    ($($lower:literal),+$(,)? target: $target:literal, $first_upper:literal$(, $($upper:literal),+)?) => {
        $(
            impl DMax<NDim<$lower>> for NDim<$target>
            {
                type Output = NDim<$target>;
            }
        )+
        impl DMax<NDim<$first_upper>> for NDim<$target>
        {
            type Output = NDim<$first_upper>;
        }
        $(
            $(
                impl DMax<NDim<$upper>> for NDim<$target>
                {
                    type Output = NDim<$upper>;
                }
            )+
        )?
        impl_max!($($lower),+, $target, target: $first_upper$(, $($upper),+)?);
    };
    // Helper syntax: zero lowers, target, at least one upper
    (target: $target:literal, $first_upper:literal, $($upper:literal),+) => {
        impl DMax<NDim<$first_upper>> for NDim<$target>
        {
            type Output = NDim<$first_upper>;
        }
        $(
            impl DMax<NDim<$upper>> for NDim<$target>
            {
                type Output = NDim<$upper>;
            }
        )+
        impl_max!($target, target: $first_upper, $($upper),+);
    };
}

impl_max!(target: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

impl<const N: usize> DMax<NDim<N>> for NDim<N>
where NDim<N>: Dimensionality
{
    type Output = Self;
}

macro_rules! impl_dimensionality {
    ($($d:literal),+) => {
        $(
            impl Dimensionality for NDim<$d>
            {
                const N: Option<usize> = Some($d);

                type Smaller = NDim<{$d - 1}>;

                type Larger = NDim<{$d + 1}>;
            }
        )+
    };
}

impl Dimensionality for D0
{
    const N: Option<usize> = Some(0);

    type Smaller = Self;

    type Larger = D1;
}

impl_dimensionality!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

impl Dimensionality for NDim<12>
{
    const N: Option<usize> = Some(12);

    type Smaller = D11;

    type Larger = DDyn;
}

/// The dynamic dimensionality.
///
/// This type captures dimensionalities that are unknown at compile-time.
/// See [`Dimensionality`] for more information.
///
/// This type does not carry any information about runtime dimensionality,
/// it just indicate that dimensionality is not known at compile-time.
/// This is done to avoid multiple sources of truth for runtime array
/// dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DDyn;

impl Dimensionality for DDyn
{
    const N: Option<usize> = None;

    type Smaller = Self;

    type Larger = Self;
}

impl DAdd<DDyn> for DDyn
{
    type Output = DDyn;
}

impl<const N: usize> DAdd<NDim<N>> for DDyn
{
    type Output = DDyn;
}

impl<const N: usize> DAdd<DDyn> for NDim<N>
{
    type Output = DDyn;
}

impl DMax<DDyn> for DDyn
{
    type Output = DDyn;
}

impl<const N: usize> DMax<NDim<N>> for DDyn
{
    type Output = DDyn;
}

impl<const N: usize> DMax<DDyn> for NDim<N>
{
    type Output = DDyn;
}
