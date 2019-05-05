// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use num_traits::Float;

/// An iterator of a sequence of geometrically spaced floats.
///
/// Iterator element type is `F`.
pub struct Geomspace<F> {
    sign: F,
    start: F,
    step: F,
    index: usize,
    len: usize,
}

impl<F> Iterator for Geomspace<F>
where
    F: Float,
{
    type Item = F;

    #[inline]
    fn next(&mut self) -> Option<F> {
        if self.index >= self.len {
            None
        } else {
            // Calculate the value just like numpy.linspace does
            let i = self.index;
            self.index += 1;
            let exponent = self.start + self.step * F::from(i).unwrap();
            Some(self.sign * exponent.exp())
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len - self.index;
        (n, Some(n))
    }
}

impl<F> DoubleEndedIterator for Geomspace<F>
where
    F: Float,
{
    #[inline]
    fn next_back(&mut self) -> Option<F> {
        if self.index >= self.len {
            None
        } else {
            // Calculate the value just like numpy.linspace does
            self.len -= 1;
            let i = self.len;
            let exponent = self.start + self.step * F::from(i).unwrap();
            Some(self.sign * exponent.exp())
        }
    }
}

impl<F> ExactSizeIterator for Geomspace<F> where Geomspace<F>: Iterator {}

/// An iterator of a sequence of geometrically spaced values.
///
/// The `Geomspace` has `n` elements, where the first element is `a` and the
/// last element is `b`.
///
/// Iterator element type is `F`, where `F` must be either `f32` or `f64`.
///
/// **Panics** if the interval `[a, b]` contains zero (including the end points).
#[inline]
pub fn geomspace<F>(a: F, b: F, n: usize) -> Geomspace<F>
where
    F: Float,
{
    assert!(
        a != F::zero() && b != F::zero(),
        "Start and/or end of geomspace cannot be zero.",
    );
    assert!(
        a.is_sign_negative() == b.is_sign_negative(),
        "Logarithmic interval cannot cross 0."
    );

    let log_a = a.abs().ln();
    let log_b = b.abs().ln();
    let step = if n > 1 {
        let nf: F = F::from(n).unwrap();
        (log_b - log_a) / (nf - F::one())
    } else {
        F::zero()
    };
    Geomspace {
        sign: a.signum(),
        start: log_a,
        step: step,
        index: 0,
        len: n,
    }
}

#[cfg(test)]
mod tests {
    use super::geomspace;

    #[test]
    #[cfg(feature = "approx")]
    fn valid() {
        use approx::assert_abs_diff_eq;
        use crate::{arr1, Array1};

        let array: Array1<_> = geomspace(1e0, 1e3, 4).collect();
        assert_abs_diff_eq!(array, arr1(&[1e0, 1e1, 1e2, 1e3]), epsilon = 1e-12);

        let array: Array1<_> = geomspace(1e3, 1e0, 4).collect();
        assert_abs_diff_eq!(array, arr1(&[1e3, 1e2, 1e1, 1e0]), epsilon = 1e-12);

        let array: Array1<_> = geomspace(-1e3, -1e0, 4).collect();
        assert_abs_diff_eq!(array, arr1(&[-1e3, -1e2, -1e1, -1e0]), epsilon = 1e-12);

        let array: Array1<_> = geomspace(-1e0, -1e3, 4).collect();
        assert_abs_diff_eq!(array, arr1(&[-1e0, -1e1, -1e2, -1e3]), epsilon = 1e-12);
    }

    #[test]
    fn iter_forward() {
        let mut iter = geomspace(1.0f64, 1e3, 4);

        assert!(iter.size_hint() == (4, Some(4)));

        assert!((iter.next().unwrap() - 1e0).abs() < 1e-5);
        assert!((iter.next().unwrap() - 1e1).abs() < 1e-5);
        assert!((iter.next().unwrap() - 1e2).abs() < 1e-5);
        assert!((iter.next().unwrap() - 1e3).abs() < 1e-5);
        assert!(iter.next().is_none());

        assert!(iter.size_hint() == (0, Some(0)));
    }

    #[test]
    fn iter_backward() {
        let mut iter = geomspace(1.0f64, 1e3, 4);

        assert!(iter.size_hint() == (4, Some(4)));

        assert!((iter.next_back().unwrap() - 1e3).abs() < 1e-5);
        assert!((iter.next_back().unwrap() - 1e2).abs() < 1e-5);
        assert!((iter.next_back().unwrap() - 1e1).abs() < 1e-5);
        assert!((iter.next_back().unwrap() - 1e0).abs() < 1e-5);
        assert!(iter.next_back().is_none());

        assert!(iter.size_hint() == (0, Some(0)));
    }

    #[test]
    #[should_panic]
    fn zero_lower() {
        geomspace(0.0, 1.0, 4);
    }

    #[test]
    #[should_panic]
    fn zero_upper() {
        geomspace(1.0, 0.0, 4);
    }

    #[test]
    #[should_panic]
    fn zero_included() {
        geomspace(-1.0, 1.0, 4);
    }
}
