// Element-wise methods for ndarray

#[cfg(feature = "std")]
use num_complex::Complex;
#[cfg(feature = "std")]
use num_traits::{Float, Zero};

use crate::imp_prelude::*;

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
    }

    /// Square (two powers) of each element.
    #[must_use = "method returns a new array and does not mutate the original value"]
    pub fn pow2(&self) -> Array<A, D>
    {
        self.mapv(|v: A| v * v)
    }
}

#[cfg(feature = "std")]
impl<A, D> ArrayRef<A, D>
where
    D: Dimension,
    A: Clone + Zero,
{
    /// Map the array into the real part of a complex array; the imaginary part is 0.
    ///
    /// # Example
    /// ```
    /// use ndarray::*;
    /// use num_complex::Complex;
    ///
    /// let arr = array![1.0, -1.0, 0.0];
    /// let complex = arr.to_complex_re();
    ///
    /// assert_eq!(complex[0], Complex::new(1.0, 0.0));
    /// assert_eq!(complex[1], Complex::new(-1.0, 0.0));
    /// assert_eq!(complex[2], Complex::new(0.0, 0.0));
    /// ```
    ///
    /// # See Also
    /// [ArrayRef::to_complex_im]
    #[must_use = "method returns a new array and does not mutate the original value"]
    pub fn to_complex_re(&self) -> Array<Complex<A>, D>
    {
        self.mapv(|v| Complex::new(v, A::zero()))
    }

    /// Map the array into the imaginary part of a complex array; the real part is 0.
    ///
    /// # Example
    /// ```
    /// use ndarray::*;
    /// use num_complex::Complex;
    ///
    /// let arr = array![1.0, -1.0, 0.0];
    /// let complex = arr.to_complex_im();
    ///
    /// assert_eq!(complex[0], Complex::new(0.0, 1.0));
    /// assert_eq!(complex[1], Complex::new(0.0, -1.0));
    /// assert_eq!(complex[2], Complex::new(0.0, 0.0));
    /// ```
    ///
    /// # See Also
    /// [ArrayRef::to_complex_re]
    #[must_use = "method returns a new array and does not mutate the original value"]
    pub fn to_complex_im(&self) -> Array<Complex<A>, D>
    {
        self.mapv(|v| Complex::new(A::zero(), v))
    }
}

/// # Angle calculation methods for arrays
///
/// Methods for calculating phase angles of complex values in arrays.
#[cfg(feature = "std")]
impl<A, D> ArrayRef<Complex<A>, D>
where
    D: Dimension,
    A: Float,
{
    /// Return the [phase angle (argument)](https://en.wikipedia.org/wiki/Argument_(complex_analysis)) of complex values in the array.
    ///
    /// This function always returns the same float type as was provided to it. Leaving the exact precision left to the user.
    /// The angles are returned in ``radians`` and in the range ``(-π, π]``.
    /// To get the angles in degrees, use the [`to_degrees()`][ArrayRef::to_degrees] method on the resulting array.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use num_complex::Complex;
    /// use std::f64::consts::PI;
    ///
    /// let complex_arr = array![
    ///     Complex::new(1.0f64, 0.0),
    ///     Complex::new(0.0, 1.0),
    ///     Complex::new(1.0, 1.0),
    /// ];
    /// let angles = complex_arr.angle();
    /// assert!((angles[0] - 0.0).abs() < 1e-10);
    /// assert!((angles[1] - PI/2.0).abs() < 1e-10);
    /// assert!((angles[2] - PI/4.0).abs() < 1e-10);
    /// ```
    #[must_use = "method returns a new array and does not mutate the original value"]
    pub fn angle(&self) -> Array<A, D>
    {
        self.mapv(|v| v.im.atan2(v.re))
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

#[cfg(all(test, feature = "std"))]
mod angle_tests
{
    use crate::Array;
    use num_complex::Complex;
    use std::f64::consts::PI;

    /// Helper macro for floating-point comparison
    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr, $tol:expr $(, $msg:expr)?) => {{
            let (a, b) = ($a, $b);
            assert!(
                (a - b).abs() < $tol,
                concat!(
                    "assertion failed: |left - right| >= tol\n",
                    " left: {left:?}\n right: {right:?}\n tol: {tol:?}\n",
                    $($msg,)?
                ),
                left = a,
                right = b,
                tol = $tol
            );
        }};
    }

    #[test]
    fn test_complex_numbers_radians()
    {
        let arr = Array::from_vec(vec![
            Complex::new(1.0f64, 0.0),    // 0
            Complex::new(0.0, 1.0),    // π/2
            Complex::new(-1.0, 0.0),   // π
            Complex::new(0.0, -1.0),   // -π/2
            Complex::new(1.0, 1.0),    // π/4
            Complex::new(-1.0, -1.0),  // -3π/4
        ]);
        let a = arr.angle();

        assert_approx_eq!(a[0], 0.0, 1e-10);
        assert_approx_eq!(a[1], PI / 2.0, 1e-10);
        assert_approx_eq!(a[2], PI, 1e-10);
        assert_approx_eq!(a[3], -PI / 2.0, 1e-10);
        assert_approx_eq!(a[4], PI / 4.0, 1e-10);
        assert_approx_eq!(a[5], -3.0 * PI / 4.0, 1e-10);
    }

    #[test]
    fn test_complex_numbers_degrees()
    {
        let arr = Array::from_vec(vec![
            Complex::new(1.0f64, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(-1.0, 0.0),
            Complex::new(1.0, 1.0),
        ]);
        let a = arr.angle().to_degrees();

        assert_approx_eq!(a[0], 0.0, 1e-10);
        assert_approx_eq!(a[1], 90.0, 1e-10);
        assert_approx_eq!(a[2], 180.0, 1e-10);
        assert_approx_eq!(a[3], 45.0, 1e-10);
    }

    #[test]
    fn test_signed_zeros()
    {
        let arr = Array::from_vec(vec![
            Complex::new(0.0f64, 0.0),
            Complex::new(-0.0, 0.0),
            Complex::new(0.0, -0.0),
            Complex::new(-0.0, -0.0),
        ]);
        let a = arr.angle();

        assert!(a[0] >= 0.0 && a[0].abs() < 1e-10);
        assert_approx_eq!(a[1], PI, 1e-10);
        assert!(a[2] <= 0.0 && a[2].abs() < 1e-10);
        assert_approx_eq!(a[3], -PI, 1e-10);
    }

    #[test]
    fn test_edge_cases()
    {
        let arr = Array::from_vec(vec![
            Complex::new(f64::INFINITY, 0.0),
            Complex::new(0.0, f64::INFINITY),
            Complex::new(f64::NEG_INFINITY, 0.0),
            Complex::new(0.0, f64::NEG_INFINITY),
        ]);
        let a = arr.angle();

        assert_approx_eq!(a[0], 0.0, 1e-10);
        assert_approx_eq!(a[1], PI / 2.0, 1e-10);
        assert_approx_eq!(a[2], PI, 1e-10);
        assert_approx_eq!(a[3], -PI / 2.0, 1e-10);
    }

    #[test]
    fn test_range_validation()
    {
        let n = 16;
        let complex_arr: Vec<_> = (0..n)
            .map(|i| {
                let theta = 2.0 * PI * (i as f64) / (n as f64);
                Complex::new(theta.cos(), theta.sin())
            })
            .collect();

        let a = Array::from_vec(complex_arr).angle();

        for &x in &a {
            assert!(x > -PI && x <= PI, "Angle {} outside (-π, π]", x);
        }
    }
}
