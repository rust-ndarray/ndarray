// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use itertools::free::enumerate;

use imp_prelude::*;
use numeric_util;

use {
    LinalgScalar,
};

use std::any::TypeId;

#[cfg(feature="blas")]
use std::cmp;
#[cfg(feature="blas")]
use std::mem::swap;
#[cfg(feature="blas")]
use std::os::raw::c_int;

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


impl<A, S> ArrayBase<S, Ix1>
    where S: Data<Elem=A>,
{
    /// Compute the dot product of one-dimensional arrays.
    ///
    /// The dot product is a sum of the elementwise products (no conjugation
    /// of complex operands, and thus not their inner product).
    ///
    /// **Panics** if the arrays are not of the same length.
    pub fn dot<S2>(&self, rhs: &ArrayBase<S2, Ix1>) -> A
        where S2: Data<Elem=A>,
              A: LinalgScalar,
    {
        self.dot_impl(rhs)
    }

    fn dot_generic<S2>(&self, rhs: &ArrayBase<S2, Ix1>) -> A
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
    fn dot_impl<S2>(&self, rhs: &ArrayBase<S2, Ix1>) -> A
        where S2: Data<Elem=A>,
              A: LinalgScalar,
    {
        self.dot_generic(rhs)
    }

    #[cfg(feature="blas")]
    fn dot_impl<S2>(&self, rhs: &ArrayBase<S2, Ix1>) -> A
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
                unsafe {
                    let (lhs_ptr, n, incx) = blas_1d_params(self.ptr,
                                                            self.len(),
                                                            self.strides()[0]);
                    let (rhs_ptr, _, incy) = blas_1d_params(rhs.ptr,
                                                            rhs.len(),
                                                            rhs.strides()[0]);
                    let ret = blas_sys::c::$func(
                        n,
                        lhs_ptr as *const $ty,
                        incx,
                        rhs_ptr as *const $ty,
                        incy);
                    return cast_as::<$ty, A>(&ret);
                }
            }
                }}
            }

            dot!{f32, cblas_sdot};
            dot!{f64, cblas_ddot};
        }
        self.dot_generic(rhs)
    }
}

/// Return a pointer to the starting element in BLAS's view.
///
/// BLAS wants a pointer to the element with lowest address,
/// which agrees with our pointer for non-negative strides, but
/// is at the opposite end for negative strides.
#[cfg(feature="blas")]
unsafe fn blas_1d_params<A>(ptr: *const A, len: usize, stride: isize)
    -> (*const A, blas_index, blas_index)
{
    // [x x x x]
    //        ^--ptr
    //        stride = -1
    //  ^--blas_ptr = ptr + (len - 1) * stride
    if stride >= 0 || len == 0 {
        (ptr, len as blas_index, stride as blas_index)
    } else {
        let ptr = ptr.offset((len - 1) as isize * stride);
        (ptr, len as blas_index, stride as blas_index)
    }
}

/// Matrix Multiplication
///
/// For two-dimensional arrays, the dot method computes the matrix
/// multiplication.
pub trait Dot<Rhs> {
    /// The result of the operation.
    ///
    /// For two-dimensional arrays: a rectangular array.
    type Output;
    fn dot(&self, rhs: &Rhs) -> Self::Output;
}

impl<A, S> ArrayBase<S, Ix2>
    where S: Data<Elem=A>,
{
    /// Perform matrix multiplication of rectangular arrays `self` and `rhs`.
    ///
    /// `Rhs` may be either a one-dimensional or a two-dimensional array.
    ///
    /// If Rhs is two-dimensional, they array shapes must agree in the way that
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
    ///     a.dot(&b) == arr2(&[[5., 8.],
    ///                         [2., 3.]])
    /// );
    /// ```
    pub fn dot<Rhs>(&self, rhs: &Rhs) -> <Self as Dot<Rhs>>::Output
        where Self: Dot<Rhs>
    {
        Dot::dot(self, rhs)
    }
}

impl<A, S, S2> Dot<ArrayBase<S2, Ix2>> for ArrayBase<S, Ix2>
    where S: Data<Elem=A>,
          S2: Data<Elem=A>,
          A: LinalgScalar,
{
    type Output = Array2<A>;
    fn dot(&self, b: &ArrayBase<S2, Ix2>) -> Array2<A>
    {
        let a = self.view();
        let b = b.view();
        let ((m, k), (k2, n)) = (a.dim_pattern(), b.dim_pattern());
        if k != k2 || m.checked_mul(n).is_none() {
            return dot_shape_error(m, k, k2, n);
        }

        let lhs_s0 = a.strides()[0];
        let rhs_s0 = b.strides()[0];
        let column_major = lhs_s0 == 1 && rhs_s0 == 1;
        // A is Copy so this is safe
        let mut v = Vec::with_capacity(m * n);
        let mut c;
        unsafe {
            v.set_len(m * n);
            c = Array::from_shape_vec_unchecked((m, n).set_f(column_major), v);
        }
        mat_mul_impl(A::one(), &a, &b, A::zero(), &mut c.view_mut());
        c
    }
}

#[cold]
#[inline(never)]
fn dot_shape_error(m: usize, k: usize, k2: usize, n: usize) -> ! {
    if m.checked_mul(n).is_none() {
        panic!("ndarray: shape {} × {} overflows type range", m, n);
    }
    panic!("ndarray: inputs {} × {} and {} × {} are not compatible for matrix multiplication",
           m, k, k2, n);
}

#[cold]
#[inline(never)]
fn general_dot_shape_error(m: usize, k: usize, k2: usize, n: usize, c1: usize, c2: usize) -> ! {
    panic!("ndarray: inputs {} × {}, {} × {}, and output {} × {} are not compatible for matrix multiplication",
           m, k, k2, n, c1, c2);
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
impl<A, S, S2> Dot<ArrayBase<S2, Ix1>> for ArrayBase<S, Ix2>
    where S: Data<Elem=A>,
          S2: Data<Elem=A>,
          A: LinalgScalar,
{
    type Output = Array<A, Ix1>;
    fn dot(&self, rhs: &ArrayBase<S2, Ix1>) -> Array<A, Ix1>
    {
        let ((m, a), n) = (self.dim_pattern(), rhs.dim_pattern());
        if a != n {
            return dot_shape_error(m, a, n, 1);
        }

        // Avoid initializing the memory in vec -- set it during iteration
        let mut res_elems = Vec::<A>::with_capacity(m as usize);
        unsafe {
            res_elems.set_len(m as usize);
        }
        for (i, rr) in enumerate(&mut res_elems) {
            unsafe {
                *rr = (0..a).fold(A::zero(),
                    move |s, k| s + *self.uget(Ix2(i, k)) * *rhs.uget(k)
                );
            }
        }
        unsafe {
            ArrayBase::from_shape_vec_unchecked(m, res_elems)
        }
    }
}

impl<A, S, D> ArrayBase<S, D>
    where S: Data<Elem=A>,
          D: Dimension,
{
    /// Perform the operation `self += alpha * rhs` efficiently, where
    /// `alpha` is a scalar and `rhs` is another array. This operation is
    /// also known as `axpy` in BLAS.
    ///
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting isn’t possible.
    pub fn scaled_add<S2, E>(&mut self, alpha: A, rhs: &ArrayBase<S2, E>)
        where S: DataMut,
              S2: Data<Elem=A>,
              A: LinalgScalar,
              E: Dimension,
    {
        self.zip_mut_with(rhs, move |y, &x| *y = *y + (alpha * x));
    }
}

// mat_mul_impl uses ArrayView arguments to send all array kinds into
// the same instantiated implementation.
#[cfg(not(feature="blas"))]
use self::mat_mul_general as mat_mul_impl;

#[cfg(feature="blas")]
fn mat_mul_impl<A>(alpha: A,
                   lhs: &ArrayView2<A>,
                   rhs: &ArrayView2<A>,
                   beta: A,
                   c: &mut ArrayViewMut2<A>)
    where A: LinalgScalar,
{
    // size cutoff for using BLAS
    let cut = GEMM_BLAS_CUTOFF;
    let ((mut m, a), (_, mut n)) = (lhs.dim_pattern(), rhs.dim_pattern());
    if !(m > cut || n > cut || a > cut) ||
        !(same_type::<A, f32>() || same_type::<A, f64>()) {
        return mat_mul_general(alpha, lhs, rhs, beta, c);
    }
    {
        // Use `c` for c-order and `f` for an f-order matrix
        // We can handle c * c, f * f generally and
        // c * f and f * c if the `f` matrix is square.
        let mut lhs_ = lhs.view();
        let mut rhs_ = rhs.view();
        let mut c_ = c.view_mut();
        let lhs_s0 = lhs_.strides()[0];
        let rhs_s0 = rhs_.strides()[0];
        let both_f = lhs_s0 == 1 && rhs_s0 == 1;
        let mut lhs_trans = CblasNoTrans;
        let mut rhs_trans = CblasNoTrans;
        if both_f {
            // A^t B^t = C^t => B A = C
            let lhs_t = lhs_.reversed_axes();
            lhs_ = rhs_.reversed_axes();
            rhs_ = lhs_t;
            c_ = c_.reversed_axes();
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
                if blas_row_major_2d::<$ty, _>(&lhs_)
                    && blas_row_major_2d::<$ty, _>(&rhs_)
                    && blas_row_major_2d::<$ty, _>(&c_)
                {
                    let (m, k) = match lhs_trans {
                        CblasNoTrans => lhs_.dim_pattern(),
                        _ => {
                            let (rows, cols) = lhs_.dim_pattern();
                            (cols, rows)
                        }
                    };
                    let n = match rhs_trans {
                        CblasNoTrans => rhs_.dim()[1],
                        _ => rhs_.dim()[0],
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
                        cast_as(&alpha),        // alpha
                        lhs_.ptr as *const _,   // a
                        lhs_stride, // lda
                        rhs_.ptr as *const _,   // b
                        rhs_stride, // ldb
                        cast_as(&beta),         // beta
                        c_.ptr as *mut _,       // c
                        c_.strides()[0] as blas_index, // ldc
                    );
                    }
                return;
                }
            }
        }
        gemm!(f32, cblas_sgemm);
        gemm!(f64, cblas_dgemm);
    }
    mat_mul_general(alpha, lhs, rhs, beta, c)
}

/// C ← α A B + β C
fn mat_mul_general<A>(alpha: A,
                      lhs: &ArrayView2<A>,
                      rhs: &ArrayView2<A>,
                      beta: A,
                      c: &mut ArrayViewMut2<A>)
    where A: LinalgScalar,
{
    let ((m, k), (_, n)) = (lhs.dim_pattern(), rhs.dim_pattern());

    // common parameters for gemm
    let ap = lhs.as_ptr();
    let bp = rhs.as_ptr();
    let cp = c.as_mut_ptr();
    let (rsc, csc) = (c.strides()[0], c.strides()[1]);
    if same_type::<A, f32>() {
        unsafe {
            ::matrixmultiply::sgemm(
                m, k, n,
                cast_as(&alpha),
                ap as *const _,
                lhs.strides()[0],
                lhs.strides()[1],
                bp as *const _,
                rhs.strides()[0],
                rhs.strides()[1],
                cast_as(&beta),
                cp as *mut _,
                rsc, csc
            );
        }
    } else if same_type::<A, f64>() {
        unsafe {
            ::matrixmultiply::dgemm(
                m, k, n,
                cast_as(&alpha),
                ap as *const _,
                lhs.strides()[0],
                lhs.strides()[1],
                bp as *const _,
                rhs.strides()[0],
                rhs.strides()[1],
                cast_as(&beta),
                cp as *mut _,
                rsc, csc
            );
        }
    } else {
        // initialize memory if beta is zero
        if beta.is_zero() {
            c.fill(beta);
        }

        let mut i = 0;
        let mut j = 0;
        loop {
            unsafe {
                let elt = c.uget_mut((i, j));
                *elt = *elt * beta + alpha * (0..k).fold(A::zero(),
                    move |s, x| s + *lhs.uget((i, x)) * *rhs.uget((x, j)));
            }
            j += 1;
            if j == n {
                j = 0;
                i += 1;
                if i == m {
                    break;
                }
            }
        }
    }
}

/// General matrix multiplication.
///
/// Compute C ← α A B + β C
///
/// The array shapes must agree in the way that
/// if `a` is *M* × *N*, then `b` is *N* × *K* and `c` is *M* × *K*.
///
/// ***Panics*** if array shapes are not compatible
pub fn general_mat_mul<A, S1, S2, S3>(alpha: A,
                                      a: &ArrayBase<S1, Ix2>,
                                      b: &ArrayBase<S2, Ix2>,
                                      beta: A,
                                      c: &mut ArrayBase<S3, Ix2>)
    where S1: Data<Elem=A>,
          S2: Data<Elem=A>,
          S3: DataMut<Elem=A>,
          A: LinalgScalar,
{
    let ((m, k), (k2, n)) = (a.dim_pattern(), b.dim_pattern());
    let (m2, n2) = c.dim_pattern();
    if k != k2 || m != m2 || n != n2 {
        return general_dot_shape_error(m, k, k2, n, m2, n2);
    }
    mat_mul_impl(alpha, &a.view(), &b.view(), beta, &mut c.view_mut());
}

#[inline(always)]
/// Return `true` if `A` and `B` are the same type
fn same_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

// Read pointer to type `A` as type `B`.
//
// **Panics** if `A` and `B` are not the same type
fn cast_as<A: 'static + Copy, B: 'static + Copy>(a: &A) -> B {
    assert!(same_type::<A, B>());
    unsafe {
        ::std::ptr::read(a as *const _ as *const B)
    }
}

#[cfg(feature="blas")]
fn blas_compat_1d<A, S>(a: &ArrayBase<S, Ix1>) -> bool
    where S: Data,
          A: 'static,
          S::Elem: 'static,
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
fn blas_row_major_2d<A, S>(a: &ArrayBase<S, Ix2>) -> bool
    where S: Data,
          A: 'static,
          S::Elem: 'static,
{
    if !same_type::<A, S::Elem>() {
        return false;
    }
    let s0 = a.strides()[0];
    let s1 = a.strides()[1];
    if s1 != 1 {
        return false;
    }
    if s0 < 1 || s1 < 1 {
        return false;
    }
    if (s0 > blas_index::max_value() as isize || s0 < blas_index::min_value() as isize) ||
        (s1 > blas_index::max_value() as isize || s1 < blas_index::min_value() as isize)
    {
        return false;
    }
    let (m, n) = a.dim_pattern();
    if m > blas_index::max_value() as usize ||
        n > blas_index::max_value() as usize
    {
        return false;
    }
    true
}
