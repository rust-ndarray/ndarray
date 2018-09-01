// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Add;
use libnum::{self, Zero, Float};
use itertools::free::enumerate;

use imp_prelude::*;
use numeric_util;

use {
    LinalgScalar,
    FoldWhile,
    Zip,
};

/// Used to choose the interpolation strategy in [`percentile_axis_mut`].
///
/// [`percentile_axis_mut`]: struct.ArrayBase.html#method.percentile_axis_mut
pub trait Interpolate<T> {
    fn needs_lower(q: f64, len: usize) -> bool;
    fn needs_upper(q: f64, len: usize) -> bool;
    fn interpolate(lower: Option<T>, upper: Option<T>, q: f64, len: usize) -> T;
}

struct Upper;
struct Lower;
struct Nearest;

impl<T> Interpolate<T> for Upper {
    fn needs_lower(q: f64, len: usize) -> bool {
        false
    }
    fn needs_upper(q: f64, len: usize) -> bool {
        true
    }
    fn interpolate(lower: Option<T>, upper: Option<T>, q: f64, len: usize) -> T {
       upper.unwrap()
    }
}

impl<T> Interpolate<T> for Lower {
    fn needs_lower(q: f64, len: usize) -> bool {
        true
    }
    fn needs_upper(q: f64, len: usize) -> bool {
        false
    }
    fn interpolate(lower: Option<T>, upper: Option<T>, q: f64, len: usize) -> T {
        lower.unwrap()
    }
}

impl<T> Interpolate<T> for Nearest {
    fn needs_lower(q: f64, len: usize) -> bool {
        let float_percentile_index = ((len - 1) as f64) * q;
        (float_percentile_index.round() - float_percentile_index) <= 0.
    }
    fn needs_upper(q: f64, len: usize) -> bool {
        !Self::needs_lower(q, len)
    }
    fn interpolate(lower: Option<T>, upper: Option<T>, q: f64, len: usize) -> T {
        if Self::needs_lower(q, len) {
            lower.unwrap()
        } else {
            upper.unwrap()
        }
    }
}

/// Numerical methods for arrays.
impl<A, S, D> ArrayBase<S, D>
    where S: Data<Elem=A>,
          D: Dimension,
{
    /// Return the sum of all elements in the array.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert_eq!(a.scalar_sum(), 10.);
    /// ```
    pub fn scalar_sum(&self) -> A
        where A: Clone + Add<Output=A> + libnum::Zero,
    {
        if let Some(slc) = self.as_slice_memory_order() {
            return numeric_util::unrolled_sum(slc);
        }
        let mut sum = A::zero();
        for row in self.inner_rows() {
            if let Some(slc) = row.as_slice() {
                sum = sum + numeric_util::unrolled_sum(slc);
            } else {
                sum = sum + row.iter().fold(A::zero(), |acc, elt| acc + elt.clone());
            }
        }
        sum
    }

    /// Return sum along `axis`.
    ///
    /// ```
    /// use ndarray::{aview0, aview1, arr2, Axis};
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert!(
    ///     a.sum_axis(Axis(0)) == aview1(&[4., 6.]) &&
    ///     a.sum_axis(Axis(1)) == aview1(&[3., 7.]) &&
    ///
    ///     a.sum_axis(Axis(0)).sum_axis(Axis(0)) == aview0(&10.)
    /// );
    /// ```
    ///
    /// **Panics** if `axis` is out of bounds.
    pub fn sum_axis(&self, axis: Axis) -> Array<A, D::Smaller>
        where A: Clone + Zero + Add<Output=A>,
              D: RemoveAxis,
    {
        let n = self.len_of(axis);
        let mut res = self.subview(axis, 0).to_owned();
        let stride = self.strides()[axis.index()];
        if self.ndim() == 2 && stride == 1 {
            // contiguous along the axis we are summing
            let ax = axis.index();
            for (i, elt) in enumerate(&mut res) {
                *elt = self.subview(Axis(1 - ax), i).scalar_sum();
            }
        } else {
            for i in 1..n {
                let view = self.subview(axis, i);
                res = res + &view;
            }
        }
        res
    }

    /// Return mean along `axis`.
    ///
    /// **Panics** if `axis` is out of bounds.
    ///
    /// ```
    /// use ndarray::{aview1, arr2, Axis};
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert!(
    ///     a.mean_axis(Axis(0)) == aview1(&[2.0, 3.0]) &&
    ///     a.mean_axis(Axis(1)) == aview1(&[1.5, 3.5])
    /// );
    /// ```
    pub fn mean_axis(&self, axis: Axis) -> Array<A, D::Smaller>
        where A: LinalgScalar,
              D: RemoveAxis,
    {
        let n = self.len_of(axis);
        let sum = self.sum_axis(axis);
        let mut cnt = A::one();
        for _ in 1..n {
            cnt = cnt + A::one();
        }
        sum / &aview0(&cnt)
    }

    /// Return the qth percentile of the data along the specified axis.
    ///
    /// `q` needs to be a float between 0 and 1, bounds included.
    /// The qth percentile for a 1-dimensional lane of length `N` is defined
    /// as the element that would be indexed as `(N-1)q` if the lane were to be sorted
    /// in increasing order.
    /// If `(N-1)q` is not an integer the desired percentile lies between
    /// two data points: we return the lower, nearest or higher datapoint depending
    /// on `interpolation_strategy`.
    ///
    /// Some examples:
    /// - `q=0.` returns the minimum along each 1-dimensional lane;
    /// - `q=0.5` returns the median along each 1-dimensional lane;
    /// - `q=1.` returns the maximum along each 1-dimensional lane.
    /// (`q=0` and `q=1` are considered improper percentiles)
    ///
    /// The array is shuffled **in place** along each 1-dimensional lane in
    /// order to produce the required percentile without allocating a copy
    /// of the original array. Each 1-dimensional lane is shuffled independently
    /// from the others.
    /// No assumptions should be made on the ordering of the array elements
    /// after this computation.
    ///
    /// Complexity ([quickselect](https://en.wikipedia.org/wiki/Quickselect)):
    /// - average case: O(`m`);
    /// - worst case: O(`m`^2);
    /// where `m` is the number of elements in the array.
    ///
    /// **Panics** if `axis` is out of bounds or if `q` is not between
    /// `0.` and `1.` (inclusive).
    pub fn percentile_axis_mut(&mut self, axis: Axis, q: f64, interpolation_strategy: InterpolationStrategy) -> Array<A, D::Smaller>
        where D: RemoveAxis,
              A: Ord + Clone + Zero,
              S: DataMut,
    {
        assert!((0. <= q) && (q <= 1.));
        let float_percentile_index = ((self.len_of(axis) - 1) as f64) * q;
        let percentile_index = match interpolation_strategy {
            InterpolationStrategy::Lower => float_percentile_index.floor() as usize,
            InterpolationStrategy::Nearest => float_percentile_index.round() as usize,
            InterpolationStrategy::Higher => float_percentile_index.ceil() as usize,
        };
        self.map_axis_mut(axis, |mut x| x.sorted_get_mut(percentile_index))
    }

    /// Return variance along `axis`.
    ///
    /// The variance is computed using the [Welford one-pass
    /// algorithm](https://www.jstor.org/stable/1266577).
    ///
    /// The parameter `ddof` specifies the "delta degrees of freedom". For
    /// example, to calculate the population variance, use `ddof = 0`, or to
    /// calculate the sample variance, use `ddof = 1`.
    ///
    /// The variance is defined as:
    ///
    /// ```text
    ///               1       n
    /// variance = ――――――――   ∑ (xᵢ - x̅)²
    ///            n - ddof  i=1
    /// ```
    ///
    /// where
    ///
    /// ```text
    ///     1   n
    /// x̅ = ―   ∑ xᵢ
    ///     n  i=1
    /// ```
    ///
    /// **Panics** if `ddof` is greater than or equal to the length of the
    /// axis, if `axis` is out of bounds, or if the length of the axis is zero.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{aview1, arr2, Axis};
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.],
    ///                [5., 6.]]);
    /// let var = a.var_axis(Axis(0), 1.);
    /// assert_eq!(var, aview1(&[4., 4.]));
    /// ```
    pub fn var_axis(&self, axis: Axis, ddof: A) -> Array<A, D::Smaller>
    where
        A: Float,
        D: RemoveAxis,
    {
        let mut count = A::zero();
        let mut mean = Array::<A, _>::zeros(self.dim.remove_axis(axis));
        let mut sum_sq = Array::<A, _>::zeros(self.dim.remove_axis(axis));
        for subview in self.axis_iter(axis) {
            count = count + A::one();
            azip!(mut mean, mut sum_sq, x (subview) in {
                let delta = x - *mean;
                *mean = *mean + delta / count;
                *sum_sq = (x - *mean).mul_add(delta, *sum_sq);
            });
        }
        if ddof >= count {
            panic!("`ddof` needs to be strictly smaller than the length \
                    of the axis you are computing the variance for!")
        } else {
            let dof = count - ddof;
            sum_sq.mapv(|s| s / dof)
        }
    }

    /// Return `true` if the arrays' elementwise differences are all within
    /// the given absolute tolerance, `false` otherwise.
    ///
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting to the same shape isn’t possible.
    pub fn all_close<S2, E>(&self, rhs: &ArrayBase<S2, E>, tol: A) -> bool
        where A: Float,
              S2: Data<Elem=A>,
              E: Dimension,
    {
        !Zip::from(self)
            .and(rhs.broadcast_unwrap(self.raw_dim()))
            .fold_while((), |_, x, y| {
                if (*x - *y).abs() <= tol {
                    FoldWhile::Continue(())
                } else {
                    FoldWhile::Done(())
                }
            }).is_done()
    }
}
