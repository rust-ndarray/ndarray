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
/// See [`NDim`] and [`DDyn`] for guidance on choosing between static and dynamic dimensionalities.
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
    + DMax<D1, Output = Self>
    + DMax<Self, Output = Self>
    + DMax<DDyn, Output = DDyn>
    + DMax<Self::Smaller, Output = Self>
    + DMax<Self::Larger, Output = Self::Larger>
    + DAdd<Self>
    + DAdd<Self::Smaller>
    + DAdd<Self::Larger>
    + DAdd<D1, Output = Self::Larger>
    + DAdd<DDyn, Output = DDyn>
{
    /// The dimensionality as a constant usize, if it's not dynamic.
    const N: Option<usize>;

    type Smaller: Dimensionality;

    type Larger: Dimensionality; // And more
}

pub trait DAdd<D>
{
    type Output: Dimensionality;
}

pub trait DMax<D>
{
    type Output: Dimensionality;
}

/// The N-dimensional static dimensionality.
///
/// This type indicates dimensionalities that are known at compile-time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NDim<const N: usize>;

pub type D0 = NDim<0>;
pub type D1 = NDim<1>;
pub type D2 = NDim<2>;
pub type D3 = NDim<3>;
pub type D4 = NDim<4>;
pub type D5 = NDim<5>;
pub type D6 = NDim<6>;
pub type D7 = NDim<7>;
pub type D8 = NDim<8>;
pub type D9 = NDim<9>;
pub type D10 = NDim<10>;
pub type D11 = NDim<11>;
pub type D12 = NDim<12>;

macro_rules! impl_add {
    ($left:literal, ($($right:literal),*), ddyn: ($($rightd:literal),*)) => {
        $(
            impl DAdd<NDim<$right>> for NDim<$left>
            {
                type Output = NDim<{$left + $right}>;
            }
        )*

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
impl_add!(0, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), ddyn: ());
impl_add!(1, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), ddyn: (12));
impl_add!(2, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), ddyn: (11, 12));
impl_add!(3, (1, 2, 3, 4, 5, 6, 7, 8, 9), ddyn: (10, 11, 12));
impl_add!(4, (1, 2, 3, 4, 5, 6, 7, 8), ddyn: (9, 10, 11, 12));
impl_add!(5, (1, 2, 3, 4, 5, 6, 7), ddyn: (8, 9, 10, 11, 12));
impl_add!(6, (1, 2, 3, 4, 5, 6), ddyn: (7, 8, 9, 10, 11, 12));
impl_add!(7, (1, 2, 3, 4, 5), ddyn: (6, 7, 8, 9, 10, 11, 12));
impl_add!(8, (1, 2, 3, 4), ddyn: (5, 6, 7, 8, 9, 10, 11, 12));
impl_add!(9, (1, 2, 3), ddyn: (4, 5, 6, 7, 8, 9, 10, 11, 12));
impl_add!(10, (1, 2), ddyn: (3, 4, 5, 6, 7, 8, 9, 10, 11, 12));
impl_add!(11, (1), ddyn: (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));
impl_add!(12, (), ddyn: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));

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

impl_max!(target: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

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

impl Dimensionality for D1
{
    const N: Option<usize> = Some(1);

    type Smaller = Self;

    type Larger = D2;
}

impl_dimensionality!(2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

impl Dimensionality for NDim<12>
{
    const N: Option<usize> = Some(12);

    type Smaller = D11;

    type Larger = DDyn;
}

/// The dynamic dimensionality.
///
/// This type indicates dimensionalities that can only be known at runtime.
/// See [`Dimensionality`] for more information.
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
