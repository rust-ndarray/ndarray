// Element-wise methods for ndarray

#[cfg(feature = "std")]
use num_traits::{Float, FloatConst, NumCast};
#[cfg(feature = "std")]
use num_complex::Complex;

use crate::imp_prelude::*;

/// Optional: precision-preserving variant (returns `F`), if you want
/// an API that keeps `f32` outputs for `f32` inputs.
///
/// - Works for `f32`/`f64` and `Complex<f32>`/`Complex<f64>`.
#[cfg(feature = "std")]
pub trait HasAngle<F: Float + FloatConst> {
    /// Return the phase angle (argument) in the same precision as the input type.
    fn to_angle(&self) -> F;
}

#[cfg(feature = "std")]
impl<F> HasAngle<F> for F
where
    F: Float + FloatConst,
{
    #[inline]
    fn to_angle(&self) -> F {
        F::zero().atan2(*self)
    }
}

#[cfg(feature = "std")]
impl<F> HasAngle<F> for Complex<F>
where
    F: Float + FloatConst,
{
    #[inline]
    fn to_angle(&self) -> F {
        self.im.atan2(self.re)
    }
}

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

/// # Angle calculation methods for arrays
///
/// Methods for calculating phase angles of complex values in arrays.
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<A, D> ArrayRef<A, D>
where
    D: Dimension,
{
    /// Return the [phase angle (argument)](https://en.wikipedia.org/wiki/Argument_(complex_analysis)) of complex values in the array.
    ///
    /// This function always returns `f64` values, regardless of input precision.
    /// The angles are returned in the range (-π, π].
    ///
    /// # Arguments
    ///
    /// * `deg` - If `true`, convert radians to degrees; if `false`, return radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use num_complex::Complex;
    /// use std::f64::consts::PI;
    ///
    /// // Real numbers
    /// let real_arr = array![1.0f64, -1.0, 0.0];
    /// let angles_rad = real_arr.angle(false);
    /// let angles_deg = real_arr.angle(true);
    /// assert!((angles_rad[0] - 0.0).abs() < 1e-10);
    /// assert!((angles_rad[1] - PI).abs() < 1e-10);
    /// assert!((angles_deg[1] - 180.0).abs() < 1e-10);
    ///
    /// // Complex numbers
    /// let complex_arr = array![
    ///     Complex::new(1.0f64, 0.0),
    ///     Complex::new(0.0, 1.0),
    ///     Complex::new(1.0, 1.0),
    /// ];
    /// let angles = complex_arr.angle(false);
    /// assert!((angles[0] - 0.0).abs() < 1e-10);
    /// assert!((angles[1] - PI/2.0).abs() < 1e-10);
    /// assert!((angles[2] - PI/4.0).abs() < 1e-10);
    /// ```
    #[must_use = "method returns a new array and does not mutate the original value"]
    pub fn angle<F: Float+ FloatConst>(&self, deg: bool) -> Array<F, D>
    where
        A: HasAngle<F>,
    {
        let mut result = self.map(|x| x.to_angle());
        if deg {
            result.mapv_inplace(|a| a *  F::from(180.0).unwrap() / F::PI());
        }
        result
    }
}

/// # Precision-preserving angle calculation methods
///
/// Methods for calculating phase angles that preserve input precision.
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<A, D> ArrayRef<A, D>
where
    D: Dimension,
{
    /// Return the [phase angle (argument)](https://en.wikipedia.org/wiki/Argument_(complex_analysis)) of values, preserving input precision.
    ///
    /// This method preserves the precision of the input:
    /// - `f32` and `Complex<f32>` inputs produce `f32` outputs
    /// - `f64` and `Complex<f64>` inputs produce `f64` outputs
    ///
    /// # Arguments
    ///
    /// * `deg` - If `true`, convert radians to degrees; if `false`, return radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use num_complex::Complex;
    ///
    /// // f32 precision preserved for complex numbers
    /// let complex_f32 = array![Complex::new(1.0f32, 1.0f32)];
    /// let angles_f32 = complex_f32.angle_preserve(false);
    /// // angles_f32 has type Array<f32, _>
    ///
    /// // f64 precision preserved for complex numbers
    /// let complex_f64 = array![Complex::new(1.0f64, 1.0f64)];
    /// let angles_f64 = complex_f64.angle_preserve(false);
    /// // angles_f64 has type Array<f64, _>
    /// ```
    #[must_use = "method returns a new array and does not mutate the original value"]
    pub fn angle_preserve<F>(&self, deg: bool) -> Array<F, D>
    where
        A: HasAngle<F>,
        F: Float + FloatConst,
    {
        let mut result = self.map(|x| x.to_angle());
        if deg {
            let factor = F::from(180.0).unwrap() / F::PI();
            result.mapv_inplace(|a| a * factor);
        }
        result
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

/// Scalar convenience function for angle calculation.
///
/// Calculate the [phase angle (argument)](https://en.wikipedia.org/wiki/Argument_(complex_analysis)) of a single complex value.
///
/// # Arguments
///
/// * `z` - A real or complex value (f32/f64, `Complex<f32>`/`Complex<f64>`).
/// * `deg` - If `true`, convert radians to degrees.
///
/// # Returns
///
/// The phase angle as `f64` in radians or degrees.
///
/// # Examples
///
/// ```
/// use num_complex::Complex;
/// use std::f64::consts::PI;
///
/// assert!((ndarray::angle_scalar(Complex::new(1.0f64, 1.0), false) - PI/4.0).abs() < 1e-10);
/// assert!((ndarray::angle_scalar(1.0f32, true) - 0.0).abs() < 1e-10);
/// assert!((ndarray::angle_scalar(-1.0f32, true) - 180.0).abs() < 1e-10);
/// ```
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub fn angle_scalar<F: Float + FloatConst, T: HasAngle<F>>(z: T, deg: bool) -> F
{
    let mut a = z.to_angle();
    if deg {

        a = a * <F as NumCast>::from(180.0).expect("180.0 is a valid f32 and f64 -- this should not fail") / F::PI();
    }
    a
}

/// Precision-preserving angle calculation function.
///
/// Calculate the phase angle of complex values while preserving input precision.
/// Unlike [`angle`], this function returns the same precision as the input:
/// - `f32` and `Complex<f32>` inputs produce `f32` outputs
/// - `f64` and `Complex<f64>` inputs produce `f64` outputs
///
/// # Arguments
///
/// * `z` - Array of real or complex values.
/// * `deg` - If `true`, convert radians to degrees.
///
/// # Returns
///
/// An `Array<F, D>` with the same shape as `z` and precision matching the input.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use num_complex::Complex;
///
/// // f32 precision preserved for complex numbers
/// let z32 = array![Complex::new(0.0f32, 1.0)];
/// let out32 = ndarray::angle(&z32, false);
/// // out32 has type Array<f32, _>
///
/// // f64 precision preserved for complex numbers
/// let z64 = array![Complex::new(0.0f64, -1.0)];
/// let out64 = ndarray::angle(&z64, false);
/// // out64 has type Array<f64, _>
/// ```
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub fn angle<A, F, S, D>(z: &ArrayBase<S, D>, deg: bool) -> Array<F, D>
where
    A: HasAngle<F>,
    F: Float + FloatConst,
    S: Data<Elem = A>,
    D: Dimension,
{
    let mut result = z.map(|x| x.to_angle());
    if deg {
        let factor = F::from(180.0).unwrap() / F::PI();
        result.mapv_inplace(|a| a * factor);
    }
    result
}

#[cfg(all(test, feature = "std"))]
mod angle_tests {
    use super::*;
    use crate::Array;
    use num_complex::Complex;
    use std::f64::consts::PI;

    #[test]
    fn test_real_numbers_radians() {
        let arr = Array::from_vec(vec![1.0f64, -1.0, 0.0]);
        let angles = arr.angle(false);

        assert!((angles[0] - 0.0).abs() < 1e-10, "angle(1.0) should be 0");
        assert!((angles[1] - PI).abs() < 1e-10, "angle(-1.0) should be π");
        assert!(angles[2].abs() < 1e-10, "angle(0.0) should be 0");
    }

    #[test]
    fn test_real_numbers_degrees() {
        let arr = Array::from_vec(vec![1.0f64, -1.0, 0.0]);
        let angles = arr.angle(true);

        assert!((angles[0] - 0.0).abs() < 1e-10, "angle(1.0) should be 0°");
        assert!((angles[1] - 180.0).abs() < 1e-10, "angle(-1.0) should be 180°");
        assert!(angles[2].abs() < 1e-10, "angle(0.0) should be 0°");
    }

    #[test]
    fn test_complex_numbers_radians() {
        let arr = Array::from_vec(vec![
            Complex::new(1.0f64, 0.0),     // 0
            Complex::new(0.0, 1.0),        // π/2
            Complex::new(-1.0, 0.0),       // π
            Complex::new(0.0, -1.0),       // -π/2
            Complex::new(1.0, 1.0),        // π/4
            Complex::new(-1.0, -1.0),      // -3π/4
        ]);
        let angles = arr.angle(false);

        assert!((angles[0] - 0.0).abs() < 1e-10, "angle(1+0i) should be 0");
        assert!((angles[1] - PI/2.0).abs() < 1e-10, "angle(0+1i) should be π/2");
        assert!((angles[2] - PI).abs() < 1e-10, "angle(-1+0i) should be π");
        assert!((angles[3] - (-PI/2.0)).abs() < 1e-10, "angle(0-1i) should be -π/2");
        assert!((angles[4] - PI/4.0).abs() < 1e-10, "angle(1+1i) should be π/4");
        assert!((angles[5] - (-3.0*PI/4.0)).abs() < 1e-10, "angle(-1-1i) should be -3π/4");
    }

    #[test]
    fn test_complex_numbers_degrees() {
        let arr = Array::from_vec(vec![
            Complex::new(1.0f64, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(-1.0, 0.0),
            Complex::new(1.0, 1.0),
        ]);
        let angles = arr.angle(true);

        assert!((angles[0] - 0.0).abs() < 1e-10, "angle(1+0i) should be 0°");
        assert!((angles[1] - 90.0).abs() < 1e-10, "angle(0+1i) should be 90°");
        assert!((angles[2] - 180.0).abs() < 1e-10, "angle(-1+0i) should be 180°");
        assert!((angles[3] - 45.0).abs() < 1e-10, "angle(1+1i) should be 45°");
    }

    #[test]
    fn test_signed_zeros() {
        let arr = Array::from_vec(vec![
            Complex::new(0.0f64, 0.0),      // +0 + 0i → +0
            Complex::new(-0.0, 0.0),        // -0 + 0i → +π
            Complex::new(0.0, -0.0),        // +0 - 0i → -0
            Complex::new(-0.0, -0.0),       // -0 - 0i → -π
        ]);
        let angles = arr.angle(false);

        assert!(angles[0] >= 0.0 && angles[0].abs() < 1e-10, "+0+0i should give +0");
        assert!((angles[1] - PI).abs() < 1e-10, "-0+0i should give +π");
        assert!(angles[2] <= 0.0 && angles[2].abs() < 1e-10, "+0-0i should give -0");
        assert!((angles[3] - (-PI)).abs() < 1e-10, "-0-0i should give -π");
    }

    #[test]
    fn test_angle_preserve_f32() {
        let arr = Array::from_vec(vec![
            Complex::new(1.0f32, 1.0),
            Complex::new(-1.0, 0.0),
        ]);
        let angles = arr.angle_preserve(false);

        assert!((angles[0] - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
        assert!((angles[1] - std::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn test_angle_preserve_f64() {
        let arr = Array::from_vec(vec![
            Complex::new(1.0f64, 1.0),
            Complex::new(-1.0, 0.0),
        ]);
        let angles = arr.angle_preserve(false);

        assert!((angles[0] - PI/4.0).abs() < 1e-10);
        assert!((angles[1] - PI).abs() < 1e-10);
    }

    #[test]
    fn test_angle_scalar_f64() {
        assert!((angle_scalar(Complex::new(1.0f64, 1.0), false) - PI/4.0).abs() < 1e-10);
        assert!((angle_scalar(1.0f64, false) - 0.0).abs() < 1e-10);
        assert!((angle_scalar(-1.0f64, false) - PI).abs() < 1e-10);
        assert!((angle_scalar(-1.0f64, true) - 180.0).abs() < 1e-10);
    }

    #[test]
    fn test_angle_scalar_f32() {
        assert!((angle_scalar(Complex::new(1.0f32, 1.0), false) - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
        assert!((angle_scalar(1.0f32, true) - 0.0).abs() < 1e-6);
        assert!((angle_scalar(-1.0f32, true) - 180.0).abs() < 1e-6);
    }

    #[test]
    fn test_angle_function() {
        let arr = Array::from_vec(vec![
            Complex::new(1.0f64, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(-1.0, 1.0),
        ]);
        let angles = angle(&arr, false);

        assert!((angles[0] - 0.0).abs() < 1e-10);
        assert!((angles[1] - PI/2.0).abs() < 1e-10);
        assert!((angles[2] - 3.0*PI/4.0).abs() < 1e-10);
    }

    #[test]
    fn test_angle_function_degrees() {
        let arr = Array::from_vec(vec![
            Complex::new(1.0f32, 1.0),
            Complex::new(-1.0, 0.0),
        ]);
        let angles = angle(&arr, true);

        assert!((angles[0] - 45.0).abs() < 1e-6);
        assert!((angles[1] - 180.0).abs() < 1e-6);
    }

    #[test]
    fn test_edge_cases() {
        let arr = Array::from_vec(vec![
            Complex::new(f64::INFINITY, 0.0),
            Complex::new(0.0, f64::INFINITY),
            Complex::new(f64::NEG_INFINITY, 0.0),
            Complex::new(0.0, f64::NEG_INFINITY),
        ]);
        let angles = arr.angle(false);

        assert!((angles[0] - 0.0).abs() < 1e-10, "angle(∞+0i) should be 0");
        assert!((angles[1] - PI/2.0).abs() < 1e-10, "angle(0+∞i) should be π/2");
        assert!((angles[2] - PI).abs() < 1e-10, "angle(-∞+0i) should be π");
        assert!((angles[3] - (-PI/2.0)).abs() < 1e-10, "angle(0-∞i) should be -π/2");
    }

    #[test]
    fn test_mixed_precision() {
        // Test that f32 and f64 can be mixed in the same operation
        let arr_f32 = Array::from_vec(vec![1.0f32, -1.0f32]);
        let angles_f32 = arr_f32.angle(false);

        let arr_f64 = Array::from_vec(vec![1.0f64, -1.0f64]);
        let angles_f64 = arr_f64.angle(false);

        // Results should be equivalent within floating point precision
        assert!((angles_f32[0] as f64 - angles_f64[0]).abs() < 1e-6);
        assert!((angles_f32[1] as f64 - angles_f64[1]).abs() < 1e-6);
    }

    #[test]
    fn test_range_validation() {
        // Generate points on the unit circle and verify angle range
        let n = 16;
        let mut complex_arr = Vec::new();

        for i in 0..n {
            let theta = 2.0 * PI * (i as f64) / (n as f64);
            if theta <= PI {
                complex_arr.push(Complex::new(theta.cos(), theta.sin()));
            } else {
                // For angles > π, we expect negative result in range (-π, 0]
                complex_arr.push(Complex::new(theta.cos(), theta.sin()));
            }
        }

        let arr = Array::from_vec(complex_arr);
        let angles = arr.angle(false);

        // All angles should be in range (-π, π]
        for &angle in angles.iter() {
            assert!(angle > -PI && angle <= PI, "Angle {} is outside range (-π, π]", angle);
        }
    }
}
