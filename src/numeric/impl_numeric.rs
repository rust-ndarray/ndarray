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

use rand::distributions::Uniform;
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
    pub fn percentile_axis_mut(&mut self, axis: Axis, q: f32) -> Array<A, D::Smaller>
        where D: RemoveAxis,
              A: Ord + Clone + Zero,
              S: DataMut,
    {
        let n = self.len_of(axis);
        let i = ((n as f32) * q).ceil() as usize;
        let mapping = |x| randomized_select(x, i);
        let mut out = Array::zeros(self.view().remove_axis(axis).raw_dim());
        azip!(mut lane (self.lanes_mut(axis)), mut out in {
            *out = mapping(lane);
        });
        out
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

fn randomized_select<A>(mut a: ArrayViewMut<A, Dim<[Ix; 1]>>, i: usize) -> A
    where A: Ord + Clone
{
    let n = a.len();
    if n == 0 {
        (&a[0]).clone()
    } else {
        let q = randomized_partition(&mut a.view_mut());
        let k = q + 1;
        if i == k {
            (&a[q]).clone()
        } else if i < k {
            randomized_select(a.slice_mut(s![0..q]), i)
        } else {
            randomized_select(a.slice_mut(s![(q+1)..n]), i - k)
        }
    }
}

fn randomized_partition<A>(a: &mut ArrayViewMut<A, Dim<[Ix; 1]>>) -> usize
    where A: Ord + Clone
{
    let n = a.len();
    let mut rng = thread_rng();
    let i: usize = Uniform::sample_single(0, n, &mut rng);
    a.swap(i, n-1);
    partition(a)
}

fn partition<A>(a: &mut ArrayViewMut<A, Dim<[Ix; 1]>>) -> usize
    where A: Ord + Clone
{
    let n = a.len();
    let x = (&a[n-1]).clone();
    let mut i = 0;
    for j in 0..n-1 {
        if a[j] <= x {
            a.swap(i, j);
            i += 1;
        }
    }
    a.swap(i, n-1);
    i
}

#[test]
fn test_partition() {
    let mut a = arr1(&[1, 3, 2, 10, 10]);
    let j = partition(&mut a.view_mut());
    assert_eq!(j, 4);
    for i in 0..j {
        assert!(a[i] <= a[j]);
    }
    let mut a = arr1(&[2, 3, 4, 1]);
    let j = partition(&mut a.view_mut());
    assert_eq!(j, 0);
    let n = a.len();
    for i in j+1..n {
        assert!(a[i] > a[j]);
    }
}

#[test]
fn test_randomized_select() {
    let a = arr1(&[1, 3, 2, 10]);
    let j = randomized_select(a.clone().view_mut(), 2);
    assert_eq!(j, 2);
    let j = randomized_select(a.clone().view_mut(), 1);
    assert_eq!(j, 1);
    let j = randomized_select(a.clone().view_mut(), 3);
    assert_eq!(j, 3);
}

#[test]
fn test_percentile_axis_mut() {
    let mut a = arr2(
        &[
        [1, 3, 2, 10],
        [2, 4, 3, 11],
        [3, 5, 6, 12]
        ]
    );
    let p = a.percentile_axis_mut(Axis(0), 0.5);
    assert!(p == a.subview(Axis(0), 1));
    let mut b = arr2(
        &[
        [1, 3, 2, 10],
        [2, 4, 3, 11],
        [3, 5, 6, 12],
        [4, 6, 7, 13]
        ]
    );
    let q = b.percentile_axis_mut(Axis(0), 0.5);
    assert!(q == b.subview(Axis(0), 1));
}
