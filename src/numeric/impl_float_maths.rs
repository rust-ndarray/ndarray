// Element-wise methods for ndarray

#[cfg(feature = "std")]
use num_traits::Float;

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
impl<A, S, D> ArrayBase<S, D>
where
    A: 'static + Float,
    S: Data<Elem = A>,
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
        /// Natural logarithm of each element.
        fn ln
        /// Base 2 logarithm of each element.
        fn log2
        /// Base 10 logarithm of each element.
        fn log10
        /// Cubic root of each element.
        fn cbrt
        /// Sine of each element (in radians).
        fn sin
        /// Cosine of each element (in radians).
        fn cos
        /// Tangent of each element (in radians).
        fn tan
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
    }

    /// Square (two powers) of each element.
    #[must_use = "method returns a new array and does not mutate the original value"]
    pub fn pow2(&self) -> Array<A, D>
    {
        self.mapv(|v: A| v * v)
    }
}

impl<A, S, D> ArrayBase<S, D>
where
    A: 'static + PartialOrd + Clone,
    S: Data<Elem = A>,
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

#[cfg(feature = "std")]
impl<A, S, D> ArrayBase<S, D>
where
    A: Float + 'static,
    S: Data<Elem = A>,
    D: RemoveAxis,
{
    /// Compute the softmax function along the specified axis.
    ///
    /// The softmax function is defined as:
    /// ```text
    /// softmax(x_i) = exp(x_i) / sum(exp(x_j) for j in axis)
    /// ```
    ///
    /// This function is usually used in machine learning to normalize the output of a neural network to a probability
    /// distribution.
    /// ```
    /// use ndarray::{array, Axis};
    ///
    /// let a = array![[1., 2., 3.], [4., 5., 6.0_f32]];
    /// let b = a.softmax(Axis(0)).mapv(|x| (x * 100.0).round() / 100.0);
    /// assert_eq!(b, array![[0.05, 0.05, 0.05], [0.95, 0.95, 0.95]]);
    /// let c = a.softmax(Axis(1)).mapv(|x| (x * 100.0).round() / 100.0);
    /// assert_eq!(c, array![[0.09, 0.24, 0.67], [0.09, 0.24, 0.67]]);
    /// ```
    ///
    /// # Arguments
    ///
    /// * `axis`: The axis along which to compute the softmax function (so every slice along the axis will sum to 1).
    pub fn softmax(&self, axis: Axis) -> Array<A, D>
    {
        let mut res = Array::uninit(self.raw_dim());
        for (arr, mut res) in self.lanes(axis).into_iter().zip(res.lanes_mut(axis)) {
            let max = arr
                .iter()
                // If we have NaN and the comparison fails, the max can be arbitrary as the sum and the whole result
                // will be NaN anyway, so we use an arbitrary ordering.
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let max = match max {
                Some(max) => *max,
                None => continue,
            };
            let mut sum = A::zero();
            for (i, x) in res.indexed_iter_mut() {
                let v = (arr[i] - max).exp();
                sum = sum + v;
                x.write(v);
            }
            for x in res.iter_mut() {
                // Safety: we wrote to every single element of the `res` array in the previous loop.
                x.write(*unsafe { x.assume_init_ref() } / sum);
            }
        }
        // Safety: we wrote to every single element of the array.
        unsafe { res.assume_init() }
    }
}

#[cfg(test)]
mod tests
{
    #[cfg(feature = "std")]
    #[test]
    fn test_softmax()
    {
        use super::*;
        use crate::array;

        let a = array![[1., 2., 3.], [4., 5., 6.0_f32]];
        let b = a.softmax(Axis(0)).mapv(|x| (x * 100.0).round() / 100.0);
        assert_eq!(b, array![[0.05, 0.05, 0.05], [0.95, 0.95, 0.95]]);
        let c = a.softmax(Axis(1)).mapv(|x| (x * 100.0).round() / 100.0);
        assert_eq!(c, array![[0.09, 0.24, 0.67], [0.09, 0.24, 0.67]]);

        #[cfg(feature = "approx")]
        {
            // examples copied from scipy softmax documentation

            use approx::assert_relative_eq;

            let x = array![[1., 0.5, 0.2, 3.], [1., -1., 7., 3.], [2., 12., 13., 3.]];

            let m = x.softmax(Axis(0));
            let y = array![[0.211942, 0.00001013, 0.00000275, 0.333333],
                [0.211942, 0.00000226, 0.00247262, 0.333333],
                [0.576117, 0.999988, 0.997525, 0.333333]];
            assert_relative_eq!(m, y, epsilon = 1e-5);

            let m = x.softmax(Axis(1));
            let y = array![[  1.05877e-01,   6.42177e-02,   4.75736e-02,   7.82332e-01],
                [  2.42746e-03,   3.28521e-04,   9.79307e-01,   1.79366e-02],
                [  1.22094e-05,   2.68929e-01,   7.31025e-01,   3.31885e-05]];
            assert_relative_eq!(m, y, epsilon = 1e-5);
        }
    }
}
