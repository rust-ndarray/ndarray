// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use num_traits::Float;

/// An iterator of a sequence of logarithmically evenly spaced floats.
///
/// Iterator element type is `F`.
pub struct Logspace<F> {
    current: F,
    last: F,
    step: F,
    index: usize,
    len: usize,
}

impl<F> Iterator for Logspace<F>
where
    F: Float,
{
    type Item = F;

    #[inline]
    fn next(&mut self) -> Option<F> {
        if self.index >= self.len {
            None
        } else {
            self.index += 1;

            let v = self.current;
            self.current = self.current * self.step;
            Some(v)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len - self.index;
        (n, Some(n))
    }
}

impl<F> DoubleEndedIterator for Logspace<F>
where
    F: Float,
{
    #[inline]
    fn next_back(&mut self) -> Option<F> {
        if self.index >= self.len {
            None
        } else {
            self.len -= 1;

            let v = self.last;
            self.last = self.last / self.step;
            Some(v)
        }
    }
}

impl<F> ExactSizeIterator for Logspace<F> where Logspace<F>: Iterator {}

/// Return an iterator of logarithmically evenly spaced floats.
///
/// The `Logspace` has `n` elements, where the first element is `a` and the last
/// element is `b`.
///
/// The sign of `a` and `b` must be the same so that the interval does not
/// include 0.
///
/// Iterator element type is `F`, where `F` must be either `f32` or `f64`.
#[inline]
pub fn logspace<F>(a: F, b: F, n: usize) -> Logspace<F>
where
    F: Float,
{
    assert!(
        a != F::zero() && b != F::zero(),
        "Start and/or end of logspace cannot be zero.",
    );
    assert!(
        a.is_sign_negative() == b.is_sign_negative(),
        "Logarithmic interval cannot cross 0."
    );

    let log_a = a.abs().ln();
    let log_b = b.abs().ln();
    let step = if n > 1 {
        let nf = F::from(n).unwrap();
        ((log_b - log_a) / (nf - F::one())).exp()
    } else {
        F::one()
    };
    Logspace {
        current: a,
        last: b,
        step: step,
        index: 0,
        len: n,
    }
}

#[cfg(test)]
mod tests {
    use super::logspace;
    use crate::{arr1, Array1};

    #[test]
    fn valid() {
        let array: Array1<_> = logspace(1e0, 1e3, 4).collect();
        assert!(array.all_close(&arr1(&[1e0, 1e1, 1e2, 1e3]), 1e-5));

        let array: Array1<_> = logspace(-1e3, -1e0, 4).collect();
        assert!(array.all_close(&arr1(&[-1e3, -1e2, -1e1, -1e0]), 1e-5));
    }

    #[test]
    fn iter_forward() {
        let mut iter = logspace(1.0f64, 1e3, 4);

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
        let mut iter = logspace(1.0f64, 1e3, 4);

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
        logspace(0.0, 1.0, 4);
    }

    #[test]
    #[should_panic]
    fn zero_upper() {
        logspace(1.0, 0.0, 4);
    }

    #[test]
    #[should_panic]
    fn zero_included() {
        logspace(-1.0, 1.0, 4);
    }
}
