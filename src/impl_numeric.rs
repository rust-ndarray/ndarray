
use std::ops::Add;
use libnum::{self, Float};

use imp_prelude::*;
use numeric_util;

use {
    LinalgScalar,
};

impl<A, S, D> ArrayBase<S, D>
    where S: Data<Elem=A>,
          D: Dimension,
{
    /// Return sum along `axis`.
    ///
    /// ```
    /// use ndarray::{aview0, aview1, arr2, Axis};
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert!(
    ///     a.sum(Axis(0)) == aview1(&[4., 6.]) &&
    ///     a.sum(Axis(1)) == aview1(&[3., 7.]) &&
    ///
    ///     a.sum(Axis(0)).sum(Axis(0)) == aview0(&10.)
    /// );
    /// ```
    ///
    /// **Panics** if `axis` is out of bounds.
    pub fn sum(&self, axis: Axis) -> OwnedArray<A, <D as RemoveAxis>::Smaller>
        where A: Clone + Add<Output=A>,
              D: RemoveAxis,
    {
        let n = self.shape()[axis.axis()];
        let mut res = self.subview(axis, 0).to_owned();
        for i in 1..n {
            let view = self.subview(axis, i);
            res.iadd(&view);
        }
        res
    }

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
        if let Some(slc) = self.as_slice() {
            return numeric_util::unrolled_sum(slc);
        }
        let mut sum = A::zero();
        for row in self.inner_iter() {
            if let Some(slc) = row.as_slice() {
                sum = sum + numeric_util::unrolled_sum(slc);
            } else {
                sum = sum + row.fold(A::zero(), |acc, elt| acc + elt.clone());
            }
        }
        sum
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
    ///     a.mean(Axis(0)) == aview1(&[2.0, 3.0]) &&
    ///     a.mean(Axis(1)) == aview1(&[1.5, 3.5])
    /// );
    /// ```
    pub fn mean(&self, axis: Axis) -> OwnedArray<A, <D as RemoveAxis>::Smaller>
        where A: LinalgScalar,
              D: RemoveAxis,
    {
        let n = self.shape()[axis.axis()];
        let mut sum = self.sum(axis);
        let one = libnum::one::<A>();
        let mut cnt = one;
        for _ in 1..n {
            cnt = cnt + one;
        }
        sum.idiv_scalar(&cnt);
        sum
    }

    /// Return `true` if the arrays' elementwise differences are all within
    /// the given absolute tolerance.<br>
    /// Return `false` otherwise, or if the shapes disagree.
    pub fn allclose<S2>(&self, rhs: &ArrayBase<S2, D>, tol: A) -> bool
        where A: Float,
              S2: Data<Elem=A>,
    {
        self.shape() == rhs.shape() &&
        self.iter().zip(rhs.iter()).all(|(x, y)| (*x - *y).abs() <= tol)
    }
}

