// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libnum::Zero;
use itertools::free::enumerate;
use std::cmp;

use imp_prelude::*;
use numeric_util;

use {
    LinalgScalar,
};

#[cfg(feature="blas")]
use std::mem::swap;
#[cfg(feature="blas")]
use std::os::raw::c_int;
#[cfg(feature="blas")]
use std::any::{Any, TypeId};

#[cfg(feature="blas")]
use blas_sys::c::{CblasNoTrans, CblasTrans, CblasRowMajor};
#[cfg(feature="blas")]
use blas_sys;

/// len of vector before we use blas
#[cfg(feature="blas")]
const DOT_BLAS_CUTOFF: usize = 32;
/// side of matrix before we use blas
#[cfg(feature="blas")]
const GEMM_BLAS_CUTOFF: usize = 7;
#[cfg(feature="blas")]
#[allow(non_camel_case_types)]
type blas_index = c_int; // blas index type

impl<A, S> ArrayBase<S, Ix>
    where S: Data<Elem=A>,
{
    /// Compute the dot product of one-dimensional arrays.
    ///
    /// The dot product is a sum of the elementwise products (no conjugation
    /// of complex operands, and thus not their inner product).
    ///
    /// **Panics** if the arrays are not of the same length.
    pub fn dot<S2>(&self, rhs: &ArrayBase<S2, Ix>) -> A
        where S2: Data<Elem=A>,
              A: LinalgScalar,
    {
        self.dot_impl(rhs)
    }

    fn dot_generic<S2>(&self, rhs: &ArrayBase<S2, Ix>) -> A
        where S2: Data<Elem=A>,
              A: LinalgScalar,
    {
        debug_assert_eq!(self.len(), rhs.len());
        assert!(self.len() == rhs.len());
        if let Some(self_s) = self.as_slice() {
            if let Some(rhs_s) = rhs.as_slice() {
                return numeric_util::unrolled_dot(self_s, rhs_s);
            }
        }
        let mut sum = A::zero();
        for i in 0..self.len() {
            unsafe {
                sum = sum.clone() + self.uget(i).clone() * rhs.uget(i).clone();
            }
        }
        sum
    }

    #[cfg(not(feature="blas"))]
    fn dot_impl<S2>(&self, rhs: &ArrayBase<S2, Ix>) -> A
        where S2: Data<Elem=A>,
              A: LinalgScalar,
    {
        self.dot_generic(rhs)
    }

    #[cfg(feature="blas")]
    fn dot_impl<S2>(&self, rhs: &ArrayBase<S2, Ix>) -> A
        where S2: Data<Elem=A>,
              A: LinalgScalar,
    {
        // Use only if the vector is large enough to be worth it
        if self.len() >= DOT_BLAS_CUTOFF {
            debug_assert_eq!(self.len(), rhs.len());
            assert!(self.len() == rhs.len());
            macro_rules! dot {
                ($ty:ty, $func:ident) => {{
            if blas_compat_1d::<$ty, _>(self) && blas_compat_1d::<$ty, _>(rhs) {
                let n = self.len() as blas_index;
                let incx = self.strides()[0] as blas_index;
                let incy = rhs.strides()[0] as blas_index;
                let ret = unsafe {
                    blas_sys::c::$func(
                        n,
                        self.ptr as *const $ty,
                        incx,
                        rhs.ptr as *const $ty,
                        incy)
                };
                return cast_as::<$ty, A>(&ret);
            }
                }}
            }

            dot!{f32, cblas_sdot};
            dot!{f64, cblas_ddot};
        }
        self.dot_generic(rhs)
    }
}


impl<A, S> ArrayBase<S, (Ix, Ix)>
    where S: Data<Elem=A>,
{
    /// Return an array view of row `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn row(&self, index: Ix) -> ArrayView<A, Ix>
    {
        self.subview(Axis(0), index)
    }

    /// Return a mutable array view of row `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn row_mut(&mut self, index: Ix) -> ArrayViewMut<A, Ix>
        where S: DataMut
    {
        self.subview_mut(Axis(0), index)
    }

    /// Return an array view of column `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn column(&self, index: Ix) -> ArrayView<A, Ix>
    {
        self.subview(Axis(1), index)
    }

    /// Return a mutable array view of column `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn column_mut(&mut self, index: Ix) -> ArrayViewMut<A, Ix>
        where S: DataMut
    {
        self.subview_mut(Axis(1), index)
    }

    /// Perform matrix multiplication of rectangular arrays `self` and `rhs`.
    ///
    /// The array shapes must agree in the way that
    /// if `self` is *M* × *N*, then `rhs` is *N* × *K*.
    ///
    /// Return a result array with shape *M* × *K*.
    ///
    /// **Panics** if shapes are incompatible.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [0., 1.]]);
    /// let b = arr2(&[[1., 2.],
    ///                [2., 3.]]);
    ///
    /// assert!(
    ///     a.mat_mul(&b) == arr2(&[[5., 8.],
    ///                             [2., 3.]])
    /// );
    /// ```
    ///
    pub fn mat_mul<S2>(&self, rhs: &ArrayBase<S2, (Ix, Ix)>) -> OwnedArray<A, (Ix, Ix)>
        where A: LinalgScalar,
              S2: Data<Elem=A>,
    {
        let rhs = rhs.view();
        let ((m, a), (b, n)) = (self.dim, rhs.dim);
        let (lhs_columns, rhs_rows) = (a, b);
        assert!(lhs_columns == rhs_rows);
        assert!(m.checked_mul(n).is_some());

        mat_mul_impl(self, &rhs)
    }

    /// Perform the matrix multiplication of the rectangular array `self` and
    /// column vector `rhs`.
    ///
    /// The array shapes must agree in the way that
    /// if `self` is *M* × *N*, then `rhs` is *N*.
    ///
    /// Return a result array with shape *M*.
    ///
    /// **Panics** if shapes are incompatible.
    pub fn mat_mul_col<S2>(&self, rhs: &ArrayBase<S2, Ix>) -> OwnedArray<A, Ix>
        where A: LinalgScalar,
              S2: Data<Elem=A>,
    {
        let ((m, a), n) = (self.dim, rhs.dim);
        let (self_columns, other_rows) = (a, n);
        assert!(self_columns == other_rows);

        // Avoid initializing the memory in vec -- set it during iteration
        let mut res_elems = Vec::<A>::with_capacity(m as usize);
        unsafe {
            res_elems.set_len(m as usize);
        }
        for (i, rr) in enumerate(&mut res_elems) {
            unsafe {
                *rr = (0..a).fold(A::zero(),
                    move |s, k| s + *self.uget((i, k)) * *rhs.uget(k)
                );
            }
        }
        unsafe {
            ArrayBase::from_vec_dim_unchecked(m, res_elems)
        }
    }
}

#[cfg(not(feature="blas"))]
use self::mat_mul_general as mat_mul_impl;

#[cfg(feature="blas")]
fn mat_mul_impl<A, S>(lhs: &ArrayBase<S, (Ix, Ix)>, rhs: &ArrayView<A, (Ix, Ix)>)
    -> OwnedArray<A, (Ix, Ix)>
    where A: LinalgScalar,
          S: Data<Elem=A>,
{
    // size cutoff for using BLAS
    let cut = GEMM_BLAS_CUTOFF;
    let ((mut m, a), (_, mut n)) = (lhs.dim, rhs.dim);
    if !(m > cut || n > cut || a > cut) ||
        !(same_type::<A, f32>() || same_type::<A, f64>()) {
        return mat_mul_general(lhs, rhs);
    }
    // Use `c` for c-order and `f` for an f-order matrix
    // We can handle c * c, f * f generally and
    // c * f and f * c if the `f` matrix is square.
    let mut lhs_ = lhs.view();
    let mut rhs_ = rhs.view();
    let lhs_s0 = lhs_.strides()[0];
    let rhs_s0 = rhs_.strides()[0];
    let both_f = lhs_s0 == 1 && rhs_s0 == 1;
    let mut lhs_trans = CblasNoTrans;
    let mut rhs_trans = CblasNoTrans;
    if both_f {
        // A^t B^t = C^t => B A = C
        lhs_ = lhs_.reversed_axes();
        rhs_ = rhs_.reversed_axes();
        swap(&mut lhs_, &mut rhs_);
        swap(&mut m, &mut n);
    } else if lhs_s0 == 1 && m == a {
        lhs_ = lhs_.reversed_axes();
        lhs_trans = CblasTrans;
    } else if rhs_s0 == 1 && a == n {
        rhs_ = rhs_.reversed_axes();
        rhs_trans = CblasTrans;
    }

    macro_rules! gemm {
        ($ty:ty, $gemm:ident) => {
        if blas_row_major_2d::<$ty, _>(&lhs_) && blas_row_major_2d::<$ty, _>(&rhs_) {
            let mut elems = Vec::<A>::with_capacity(m * n);
            let c;
            unsafe {
                elems.set_len(m * n);
                c = OwnedArray::from_vec_dim_unchecked((m, n), elems);
            }
            {
                let (m, k) = match lhs_trans {
                    CblasNoTrans => lhs_.dim(),
                    _ => {
                        let (rows, cols) = lhs_.dim();
                        (cols, rows)
                    }
                };
                let n = match rhs_trans {
                    CblasNoTrans => rhs_.dim().1,
                    _ => rhs_.dim().0,
                };
                // adjust strides, these may [1, 1] for column matrices
                let lhs_stride = cmp::max(lhs_.strides()[0] as blas_index, k as blas_index);
                let rhs_stride = cmp::max(rhs_.strides()[0] as blas_index, n as blas_index);

                // gemm is C ← αA^Op B^Op + βC
                // Where Op is notrans/trans/conjtrans
                unsafe {
                    blas_sys::c::$gemm(
                    CblasRowMajor,
                    lhs_trans,
                    rhs_trans,
                    m as blas_index, // m, rows of Op(a)
                    n as blas_index, // n, cols of Op(b)
                    k as blas_index, // k, cols of Op(a)
                    1.0,                  // alpha
                    lhs_.ptr as *const _, // a
                    lhs_stride, // lda
                    rhs_.ptr as *const _, // b
                    rhs_stride, // ldb
                    0.0,                   // beta
                    c.ptr as *mut _,       // c
                    c.strides()[0] as blas_index, // ldc
                );
                }
            }
            return if both_f {
                c.reversed_axes()
            } else {
                c
            };
        }
        }
    }
    gemm!(f32, cblas_sgemm);
    gemm!(f64, cblas_dgemm);
    return mat_mul_general(lhs, rhs);
}

fn mat_mul_general<A, S>(lhs: &ArrayBase<S, (Ix, Ix)>, rhs: &ArrayView<A, (Ix, Ix)>)
    -> OwnedArray<A, (Ix, Ix)>
    where A: LinalgScalar,
          S: Data<Elem=A>,
{
    let ((m, a), (_, n)) = (lhs.dim, rhs.dim);

    let lhs_s0 = lhs.strides()[0];
    let rhs_s0 = rhs.strides()[0];
    let column_major = lhs_s0 == 1 && rhs_s0 == 1;

    // Avoid initializing the memory in vec -- set it during iteration
    // Panic safe because A: Copy
    let mut res_elems = Vec::<A>::with_capacity(m as usize * n as usize);
    unsafe {
        res_elems.set_len(m as usize * n as usize);
    }
    let mut i = 0;
    let mut j = 0;
    for rr in &mut res_elems {
        unsafe {
            *rr = (0..a).fold(A::zero(),
                move |s, k| s + *lhs.uget((i, k)) * *rhs.uget((k, j)));
        }
        if !column_major {
            j += 1;
            if j == n {
                j = 0;
                i += 1;
            }
        } else {
            i += 1;
            if i == m {
                i = 0;
                j += 1;
            }
        }
    }
    unsafe {
        if !column_major {
            ArrayBase::from_vec_dim_unchecked((m, n), res_elems)
        } else {
            ArrayBase::from_vec_dim_unchecked_f((m, n), res_elems)
        }
    }
}

#[cfg(feature="blas")]
#[inline(always)]
/// Return `true` if `A` and `B` are the same type
fn same_type<A: Any, B: Any>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

#[cfg(feature="blas")]
// Read pointer to type `A` as type `B`.
//
// **Panics** if `A` and `B` are not the same type
fn cast_as<A: Any + Copy, B: Any + Copy>(a: &A) -> B {
    assert!(same_type::<A, B>());
    unsafe {
        ::std::ptr::read(a as *const _ as *const B)
    }
}

#[cfg(feature="blas")]
fn blas_compat_1d<A, S>(a: &ArrayBase<S, Ix>) -> bool
    where S: Data,
          A: Any,
          S::Elem: Any,
{
    if !same_type::<A, S::Elem>() {
        return false;
    }
    if a.len() > blas_index::max_value() as usize {
        return false;
    }
    let stride = a.strides()[0];
    if stride > blas_index::max_value() as isize ||
        stride < blas_index::min_value() as isize {
        return false;
    }
    true
}

#[cfg(feature="blas")]
fn blas_row_major_2d<A, S>(a: &ArrayBase<S, (Ix, Ix)>) -> bool
    where S: Data,
          A: Any,
          S::Elem: Any,
{
    if !same_type::<A, S::Elem>() {
        return false;
    }
    let s0 = a.strides()[0];
    let s1 = a.strides()[1];
    if s1 != 1 {
        return false;
    }
    if (s0 > blas_index::max_value() as isize || s0 < blas_index::min_value() as isize) ||
        (s1 > blas_index::max_value() as isize || s1 < blas_index::min_value() as isize)
    {
        return false;
    }
    let (m, n) = a.dim();
    if m > blas_index::max_value() as usize ||
        n > blas_index::max_value() as usize
    {
        return false;
    }
    true
}
