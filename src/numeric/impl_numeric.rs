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

use rand::prelude::*;
use rand::thread_rng;

use {
    LinalgScalar,
    FoldWhile,
    Zip,
};

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
    /// `q` needs to be a float between 0 and 1, bounds included.
    /// The qth percentile for a 1-dimensional lane of length N is defined
    /// as the element that would be indexed as `Nq` if the lane were to be sorted
    /// in increasing order. If `Nq` is not an integer the desired percentile
    /// lies between two data points: we always return the smaller data point.
    ///
    /// Some examples:
    /// - `q=0.` returns the minimum along each 1-dimensional lane;
    /// - `q=0.5` returns the median along each 1-dimensional lane;
    /// - `q=1.` returns the maximum along each 1-dimensional lane.
    /// (`q=0` and `q=1` are considered improper percentiles)
    ///
    /// The array is shuffled **in place** along each 1-dimensional lane in
    /// order to produce the required percentile without allocating a copy
    /// of the original array. Each 1-dimensional lane is shuffled indipendently
    /// from the others.
    /// No assumptions should be made on the ordering of the array elements
    /// after this computation.
    ///
    /// Complexity ([quickselect](https://en.wikipedia.org/wiki/Quickselect)):
    /// - average case: O(m);
    /// - worst case: O(m^2);
    /// where m is the number of elements in the array.
    ///
    /// **Panics** if `axis` is out of bounds, if `q` is strictly smaller
    /// than 0 or strictly bigger than 1.
    pub fn percentile_axis_mut(&mut self, axis: Axis, q: f32) -> Array<A, D::Smaller>
        where D: RemoveAxis,
              A: Ord + Clone + Zero,
              S: DataMut,
    {
        assert!((0. <= q) && (q <= 1.));
        let n = self.len_of(axis);
        let i = ((n as f32) * q).ceil() as usize;
        let mapping = |x| ith_mut(x, if i == 0 {0} else {i-1});
        let mut out = Array::zeros(self.view().remove_axis(axis).raw_dim());
        azip!(mut lane (self.lanes_mut(axis)), mut out in {
            *out = mapping(lane);
        });
        out
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
    /// **Panics** if `ddof` is greater equal than the length of `axis`.
    /// **Panics** if `axis` is out of bounds or if length of `axis` is zero.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{aview1, arr2, Axis};
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// let var = a.var_axis(Axis(0), 0.);
    /// assert_eq!(var, aview1(&[1., 1.]));
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
                *sum_sq = *sum_sq + delta * (x - *mean);
            });
        }
        if ddof >= count {
            panic!("Ddof needs to be strictly smaller than the length \
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

/// Return the `i`-th element of `a` if `a` were to be a 1-dimensional
/// array sorted in increasing order.
///
/// `a` is shuffled **in place** to retrieve the desired element:
/// no copy of the array is allocated.
/// No assumptions should be made on the ordering of `a` elements
/// after this computation.
///
/// Complexity ([quickselect](https://en.wikipedia.org/wiki/Quickselect)):
/// - average case: O(n);
/// - worst case: O(n^2);
/// where n is the number of elements in `a`.
///
/// **Panics** if `i` is greater than or equal to n.
fn ith_mut<A>(mut a: ArrayViewMut1<A>, i: usize) -> A
    where A: Ord + Clone
{
    let n = a.len();
    if n == 1 {
        (&a[0]).clone()
    } else {
        let pivot_index = random_pivot(n);
        let partition_index = partition_mut(&mut a.view_mut(), pivot_index);
        if i == partition_index {
            (&a[partition_index]).clone()
        } else if i < partition_index {
            ith_mut(a.slice_mut(s![0..partition_index]), i)
        } else {
            ith_mut(a.slice_mut(s![(partition_index+1)..n]), i - partition_index - 1)
        }
    }
}

fn random_pivot(n: usize) -> usize
{
    let mut rng = thread_rng();
    rng.gen_range(0, n)
}

/// Return the index of `a[partition_index`]` if `a` were to be sorted
/// in increasing order.
/// `a` elements are rearranged in such a way that `a[partition_index]`
/// is in the position it would be in an array sorted in increasing order.
/// All elements smaller than `a[partition_index]` are moved to its
/// left and all elements equal or greater than `a[partition_index]`
/// are moved to its right.
/// The ordering of the elements in the two partitions is undefined.
///
/// `a` is shuffled **in place** to operate the desired partition:
/// no copy of the array is allocated.
///
/// Complexity: O(n), where n is the number of elements in `a`.
///
/// **Panics** if `partition_index` is greater than or equal to n.
fn partition_mut<A>(a: &mut ArrayViewMut1<A>, partition_index: usize) -> usize
    where A: Ord + Clone
{
    let n = a.len();
    let partition_value = (&a[partition_index]).clone();
    a.swap(partition_index, n-1);
    let mut partition_boundary_index = 0;
    for j in 0..n-1 {
        if a[j] < partition_value {
            a.swap(partition_boundary_index, j);
            partition_boundary_index += 1;
        }
    }
    a.swap(partition_boundary_index, n-1);
    partition_boundary_index
}
