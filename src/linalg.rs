#![allow(non_snake_case, deprecated)]
#![cfg_attr(has_deprecated, deprecated(note="`linalg` is not in good shape."))]

//! ***Deprecated: linalg is not in good shape.***
//!
//! A few linear algebra operations on two-dimensional arrays.

use libnum::{Num, zero, Zero, One};
use libnum::Float;
use libnum::Complex;
use std::ops::{Add, Sub, Mul, Div};

use super::{Array, Ix};

/// Column vector.
pub type Col<A> = Array<A, Ix>;
/// Rectangular matrix.
pub type Mat<A> = Array<A, (Ix, Ix)>;

/// Trait union for a ring with 1.
pub trait Ring : Clone + Zero + Add<Output=Self> + Sub<Output=Self>
    + One + Mul<Output=Self> { }
impl<A: Clone + Zero + Add<Output=A> + Sub<Output=A> + One + Mul<Output=A>> Ring for A { }

/// Trait union for a field.
pub trait Field : Ring + Div<Output=Self> { }
impl<A: Ring + Div<Output=A>> Field for A { }

/// A real or complex number.
pub trait ComplexField : Copy + Field
{
    #[inline]
    fn conjugate(self) -> Self { self }
    fn sqrt_real(self) -> Self;
    #[inline]
    fn is_complex() -> bool { false }
}

impl ComplexField for f32
{
    #[inline]
    fn sqrt_real(self) -> f32 { self.sqrt() }
}

impl ComplexField for f64
{
    #[inline]
    fn sqrt_real(self) -> f64 { self.sqrt() }
}

impl<A: Num + Float> ComplexField for Complex<A>
{
    #[inline]
    fn conjugate(self) -> Complex<A> { self.conj() }
    fn sqrt_real(self) -> Complex<A> { Complex::new(self.re.sqrt(), zero()) }
    #[inline]
    fn is_complex() -> bool { true }
}

/// Return the identity matrix of dimension *n*.
///
/// Deprecated: Use `Array::eye(n)` instead.
#[cfg_attr(has_deprecated, deprecated(note="Use Array::eye instead."))]
pub fn eye<A: Clone + Zero + One>(n: Ix) -> Mat<A> {
    Array::eye(n)
}

/*
/// Return the inverse matrix of square matrix `a`.
pub fn inverse<A: Primitive>(a: &Mat<A>) -> Mat<A>
{
    fail!()
}
*/

/// Solve *a x = b* with linear least squares approximation.
///
/// It is used to find the best fit for an overdetermined system,
/// i.e. the number of rows in *a* is larger than the number of
/// unknowns *x*.
///
/// Return best fit for *x*.
pub fn least_squares<A: ComplexField>(a: &Mat<A>, b: &Col<A>) -> Col<A>
{
    // Using transpose: a.T a x = a.T b;
    // a.T a being square gives naive solution
    // x_lstsq = inv(a.T a) a.T b
    //
    // Solve using cholesky decomposition
    // aT a x = aT b
    //
    // Factor aT a into L L.T
    //
    // L L.T x = aT b
    //
    // => L z = aT b 
    //  fw subst for z
    // => L.T x = z
    //  bw subst for x estimate
    // 
    let mut aT = a.clone();
    aT.swap_axes(0, 1);
    if <A as ComplexField>::is_complex() {
        // conjugate transpose
        for elt in aT.iter_mut() {
            *elt = elt.conjugate();
        }
    }

    let aT_a = aT.mat_mul(a).into_shared();
    let mut L = cholesky(aT_a);
    let rhs = aT.mat_mul_col(b).into_shared();

    // Solve L z = aT b
    let z = subst_fw(&L, &rhs);

    // Solve L.T x = z
    if <A as ComplexField>::is_complex() {
        // conjugate transpose
        // only elements below the diagonal have imag part
        let (m, _) = L.dim();
        for i in 1..m {
            for j in 0..i {
                let elt = &mut L[[i, j]];
                *elt = elt.conjugate();
            }
        }
    }
    L.swap_axes(0, 1);

    // => x_lstsq
    subst_bw(&L, &z)
}

/// Factor *a = L L<sup>T</sup>*.
///
/// *a* should be a square matrix, hermitian and positive definite.
///
/// https://en.wikipedia.org/wiki/Cholesky_decomposition
///
/// “The Cholesky decomposition is mainly used for the numerical solution of
/// linear equations Ax = b.
///
/// If A is symmetric and positive definite, then we can solve Ax = b by first
/// computing the Cholesky decomposition A = LL*, then solving Ly = b for y by
/// forward substitution, and finally solving L*x = y for x by back
/// substitution.”
///
/// Return L.
pub fn cholesky<A: ComplexField>(a: Mat<A>) -> Mat<A>
{
    let z = zero::<A>();
    let (m, n) = a.dim();
    assert!(m == n);
    // Perform the operation in-place on `a`
    let mut L = a;
    for i in 0..m {
        // Entries 0 .. i before the diagonal
        for j in 0..i {
            // A = (
            // L²_1,1
            // L_2,1 L_1,1  L²_2,1 + L²_2,2
            // L_3,1 L_1,1  L_3,1 L_2,1 + L_3,2 L_2,2  L²_3,1 + L²_3,2 + L²_3,3
            // .. )
            let mut lik_ljk_sum = z;
            {
                // L_ik for k = 0 .. j
                // L_jk for k = 0 .. j
                let Lik = L.row(i).into_iter();
                let Ljk = L.row(j).into_iter();
                for (&lik, &ljk) in Lik.zip(Ljk).take(j as usize) {
                    lik_ljk_sum = lik_ljk_sum + lik * ljk.conjugate();
                }
            }

            // L_ij = [ A_ij - Sum(k = 1 .. j) L_ik L_jk ] / L_jj
            L[[i, j]] = (L[[i, j]] - lik_ljk_sum) / L[[j, j]];
        }

        // Diagonal where i == j
        // L_jj = Sqrt[ A_jj - Sum(k = 1 .. j) L_jk L_jk ]
        let j = i;
        let mut ljk_sum = z;
        // L_jk for k = 0 .. j
        for &ljk in L.row(j).into_iter().take(j as usize) {
            ljk_sum = ljk_sum + ljk * ljk.conjugate();
        }
        L[[j, j]] = (L[[j, j]] - ljk_sum).sqrt_real();

        // After the diagonal
        // L_ij = 0 for j > i
        for j in i + 1..n {
            L[[i, j]] = z;
        }
    }
    L
}

/// Solve *L x = b* where *L* is a lower triangular matrix.
pub fn subst_fw<A: Copy + Field>(l: &Mat<A>, b: &Col<A>) -> Col<A>
{
    let (m, n) = l.dim();
    assert!(m == n);
    assert!(m == b.dim());
    let mut x = Col::zeros(m);
    for i in 0..m {
        // b_lx_sum = b[i] - Sum(for j = 0 .. i) L_ij x_j
        let mut b_lx_sum = b[i];
        for j in 0..i {
            b_lx_sum = b_lx_sum - l[[i, j]] * x[j];
        }
        x[i] = b_lx_sum / l[[i, i]];
    }
    x
}

/// Solve *U x = b* where *U* is an upper triangular matrix.
pub fn subst_bw<A: Copy + Field>(u: &Mat<A>, b: &Col<A>) -> Col<A>
{
    let (m, n) = u.dim();
    assert!(m == n);
    assert!(m == b.dim());
    let mut x = Col::zeros(m);
    for i in (0..m).rev() {
        // b_ux_sum = b[i] - Sum(for j = i .. m) U_ij x_j
        let mut b_ux_sum = b[i];
        for j in i..m {
            b_ux_sum = b_ux_sum - u[[i, j]] * x[j];
        }
        x[i] = b_ux_sum / u[[i, i]];
    }
    x
}
