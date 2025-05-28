// Element-wise methods for ndarray

#[cfg(feature = "std")]
use num_traits::Float;
use std::borrow::Borrow;

use crate::imp_prelude::*;
use crate::IntoNdProducer;

#[cfg(feature = "std")]
macro_rules! boolean_ops {
    ($(#[$meta1:meta])* fn $func:ident
    $(#[$meta2:meta])* fn $all:ident
    $(#[$meta3:meta])* fn $any:ident) => {
        $(#[$meta1])*
        #[must_use = "method returns a new array and does not mutate the original value"]
        pub fn $func(&self) -> Array<bool, D> {
            self.mapv(A::$func)
        }
        $(#[$meta2])*
        #[must_use = "method returns a new boolean value and does not mutate the original value"]
        pub fn $all(&self) -> bool {
            $crate::Zip::from(self).all(|&elt| !elt.$func())
        }
        $(#[$meta3])*
        #[must_use = "method returns a new boolean value and does not mutate the original value"]
        pub fn $any(&self) -> bool {
            !self.$all()
        }
    };
}

#[cfg(feature = "std")]
macro_rules! unary_ops {
    ($($(#[$meta:meta])* fn $id:ident)+) => {
        $($(#[$meta])*
        #[must_use = "method returns a new array and does not mutate the original value"]
        pub fn $id(&self) -> Array<A, D> {
            self.mapv(A::$id)
        })+
    };
}

#[cfg(feature = "std")]
macro_rules! binary_ops {
    ($($(#[$meta:meta])* fn $id:ident($ty:ty))+) => {
        $($(#[$meta])*
        #[must_use = "method returns a new array and does not mutate the original value"]
        pub fn $id(&self, rhs: $ty) -> Array<A, D> {
            self.mapv(|v| A::$id(v, rhs))
        })+
    };
}

#[cfg(feature = "std")]
macro_rules! binary_ops_array {
    ($($(#[$meta:meta])* fn $fn_use:ident($ty:ty) as $fn_name:ident)+) => {
        $($(#[$meta])*
        #[must_use = "method returns a new array and does not mutate the original value"]
        pub fn $fn_name<P>(&self, rhs: P) -> Array<A, D>
        where P: IntoNdProducer<Dim = D, Item: Borrow<$ty>> {
            $crate::Zip::from(self).and(rhs).map_collect(|a, b| A::$fn_use(*a, *b.borrow()))
        })+
    };
}

/// # Element-wise methods for float arrays
///
/// Element-wise math functions for any array type that contains float number.
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<A, D> ArrayRef<A, D>
where
    A: 'static + Float,
    D: Dimension,
{
    boolean_ops! {
        /// If the number is `NaN` (not a number), then `true` is returned for each element.
        fn is_nan
        /// Return `true` if all elements are `NaN` (not a number).
        fn is_all_nan
        /// Return `true` if any element is `NaN` (not a number).
        fn is_any_nan
    }
    boolean_ops! {
        /// If the number is infinity, then `true` is returned for each element.
        fn is_infinite
        /// Return `true` if all elements are infinity.
        fn is_all_infinite
        /// Return `true` if any element is infinity.
        fn is_any_infinite
    }
    unary_ops! {
        /// The largest integer less than or equal to each element.
        fn floor
        /// The smallest integer less than or equal to each element.
        fn ceil
        /// The nearest integer of each element.
        fn round
        /// The integer part of each element.
        fn trunc
        /// The fractional part of each element.
        fn fract
        /// Absolute of each element.
        fn abs
        /// Sign number of each element.
        ///
        /// + `1.0` for all positive numbers.
        /// + `-1.0` for all negative numbers.
        /// + `NaN` for all `NaN` (not a number).
        fn signum
        /// The reciprocal (inverse) of each element, `1/x`.
        fn recip
        /// Square root of each element.
        fn sqrt
        /// `e^x` of each element (exponential function).
        fn exp
        /// `2^x` of each element.
        fn exp2
        /// `e^x - 1` of each element.
        fn exp_m1
        /// Natural logarithm of each element.
        fn ln
        /// Base 2 logarithm of each element.
        fn log2
        /// Base 10 logarithm of each element.
        fn log10
        /// `ln(1 + x)` of each element.
        fn ln_1p
        /// Cubic root of each element.
        fn cbrt
        /// Sine of each element (in radians).
        fn sin
        /// Cosine of each element (in radians).
        fn cos
        /// Tangent of each element (in radians).
        fn tan
        /// Arcsine of each element (return in radians).
        fn asin
        /// Arccosine of each element (return in radians).
        fn acos
        /// Arctangent of each element (return in radians).
        fn atan
        /// Hyperbolic sine of each element.
        fn sinh
        /// Hyperbolic cosine of each element.
        fn cosh
        /// Hyperbolic tangent of each element.
        fn tanh
        /// Inverse hyperbolic sine of each element.
        fn asinh
        /// Inverse hyperbolic cosine of each element.
        fn acosh
        /// Inverse hyperbolic tangent of each element.
        fn atanh
        /// Converts radians to degrees for each element.
        fn to_degrees
        /// Converts degrees to radians for each element.
        fn to_radians
    }
    binary_ops! {
        /// Integer power of each element.
        ///
        /// This function is generally faster than using float power.
        fn powi(i32)
        /// Float power of each element.
        fn powf(A)
        /// Logarithm of each element with respect to an arbitrary base.
        fn log(A)
        /// The positive difference between given number and each element.
        fn abs_sub(A)
        /// Length of the hypotenuse of a right-angle triangle of each element
        fn hypot(A)
        /// Four quadrant arctangent of each element in radians.
        fn atan2(A)
    }
    binary_ops_array! {
        /// Integer power of each element.
        fn powi(i32) as powi_all
        /// Float power of each element.
        fn powf(A) as powf_all
        /// Logarithm of each element with respect to an arbitrary base.
        fn log(A) as log_all
        /// The positive difference between given number and each element.
        fn abs_sub(A) as abs_sub_all
        /// Length of the hypotenuse of a right-angle triangle of each element
        fn hypot(A) as hypot_all
        /// Four quadrant arctangent of each element in radians.
        fn atan2(A) as atan2_all
    }

    /// Square (two powers) of each element.
    #[must_use = "method returns a new array and does not mutate the original value"]
    pub fn pow2(&self) -> Array<A, D>
    {
        self.mapv(|v: A| v * v)
    }
}

impl<A, D> ArrayRef<A, D>
where
    A: 'static + PartialOrd + Clone,
    D: Dimension,
{
    /// Limit the values for each element, similar to NumPy's `clip` function.
    ///
    /// ```
    /// use ndarray::array;
    ///
    /// let a = array![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
    /// assert_eq!(a.clamp(1., 8.), array![1., 1., 2., 3., 4., 5., 6., 7., 8., 8.]);
    /// assert_eq!(a.clamp(3., 6.), array![3., 3., 3., 3., 4., 5., 6., 6., 6., 6.]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `!(min <= max)`.
    pub fn clamp(&self, min: A, max: A) -> Array<A, D>
    {
        assert!(min <= max, "min must be less than or equal to max");
        self.mapv(|a| num_traits::clamp(a, min.clone(), max.clone()))
    }
}
