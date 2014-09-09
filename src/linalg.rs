#![allow(non_snake_case)]

//! A few linear algebra operations on two-dimensional arrays.

use std::num::{zero, one, Zero, One};
#[cfg(not(nocomplex))]
use libnum::Complex;

use super::{Array, Ix};

/// Column vector.
pub type Col<A> = Array<A, Ix>;
/// Rectangular matrix.
pub type Mat<A> = Array<A, (Ix, Ix)>;

/// Trait union for a ring with 1.
pub trait Ring : Clone + Zero + Add<Self, Self> + Sub<Self, Self>
    + One + Mul<Self, Self> { }
impl<A: Clone + Zero + Add<A, A> + Sub<A, A> + One + Mul<A, A>> Ring for A { }

/// Trait union for a field.
pub trait Field : Ring + Div<Self, Self> { }
impl<A: Ring + Div<A, A>> Field for A { }

/// A real or complex number.
pub trait ComplexField : Copy + Field
{
    #[inline]
    fn conjugate(self) -> Self { self }
    fn sqrt_real(self) -> Self;
    #[inline]
    fn is_complex(_mark: Option<Self>) -> bool { false }
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

#[cfg(not(nocomplex))]
impl<A: Float> ComplexField for Complex<A>
{
    #[inline]
    fn conjugate(self) -> Complex<A> { self.conj() }
    fn sqrt_real(self) -> Complex<A> { Complex::new(self.re.sqrt(), zero()) }
    #[inline]
    fn is_complex(_mark: Option<Complex<A>>) -> bool { true }
}

/// Return the identity matrix of dimension *n*.
pub fn eye<A: Clone + Zero + One>(n: Ix) -> Mat<A>
{
    let mut eye = Array::zeros((n, n));
    for a_ii in eye.diag_iter_mut() {
        *a_ii = one::<A>();
    }
    eye
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
    if ComplexField::is_complex(None::<A>) {
        // conjugate transpose
        for elt in aT.iter_mut() {
            *elt = elt.conjugate();
        }
    }

    let aT_a = aT.mat_mul(a);
    let mut L = cholesky(aT_a);
    let rhs = aT.mat_mul_col(b);

    // Solve L z = aT b
    let z = subst_fw(&L, &rhs);

    // Solve L.T x = z
    if ComplexField::is_complex(None::<A>) {
        // conjugate transpose
        // only elements below the diagonal have imag part
        let (m, _) = L.dim();
        for i in range(1, m) {
            for j in range(0, i) {
                let elt = &mut L[(i, j)];
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
    for i in range(0, m) {
        // Entries 0 .. i before the diagonal
        for j in range(0, i) {
            // A = (
            // L²_1,1
            // L_2,1 L_1,1  L²_2,1 + L²_2,2
            // L_3,1 L_1,1  L_3,1 L_2,1 + L_3,2 L_2,2  L²_3,1 + L²_3,2 + L²_3,3
            // .. )
            let mut lik_ljk_sum = z.clone();
            {
                // L_ik for k = 0 .. j
                // L_jk for k = 0 .. j
                let Lik = L.row_iter(i);
                let Ljk = L.row_iter(j);
                for (&lik, &ljk) in Lik.zip(Ljk).take(j as uint) {
                    lik_ljk_sum = lik_ljk_sum + lik * ljk.conjugate();
                }
            }

            // L_ij = [ A_ij - Sum(k = 1 .. j) L_ik L_jk ] / L_jj
            L[(i, j)] = (L[(i, j)] - lik_ljk_sum) / L[(j, j)];
        }

        // Diagonal where i == j
        // L_jj = Sqrt[ A_jj - Sum(k = 1 .. j) L_jk L_jk ]
        let j = i;
        let mut ljk_sum = z.clone();
        // L_jk for k = 0 .. j
        for &ljk in L.row_iter(j).take(j as uint) {
            ljk_sum = ljk_sum + ljk * ljk.conjugate();
        }
        L[(j, j)] = (L[(j, j)] - ljk_sum).sqrt_real();

        // After the diagonal
        // L_ij = 0 for j > i
        for j in range(i + 1, n) {
            L[(i, j)] = z.clone();
        }
    }
    L
}

/// Solve *L x = b* where *L* is a lower triangular matrix.
pub fn subst_fw<A: Field>(l: &Mat<A>, b: &Col<A>) -> Col<A>
{
    let (m, n) = l.dim();
    assert!(m == n);
    assert!(m == b.dim());
    let mut x = Vec::from_elem(m as uint, zero::<A>());
    for (i, bi) in b.indexed_iter() {
        // b_lx_sum = b[i] - Sum(for j = 0 .. i) L_ij x_j
        let mut b_lx_sum = bi.clone();
        for (lij, xj) in l.row_iter(i).zip(x.iter()).take(i as uint) {
            b_lx_sum = b_lx_sum - (*lij) * (*xj)
        }
        x.as_mut_slice()[i as uint] = b_lx_sum / l[(i, i)];
    }
    Array::from_vec(x)
}

/// Solve *U x = b* where *U* is an upper triangular matrix.
pub fn subst_bw<A: Field>(u: &Mat<A>, b: &Col<A>) -> Col<A>
{
    let (m, n) = u.dim();
    assert!(m == n);
    assert!(m == b.dim());
    let mut x = Vec::from_elem(m as uint, zero::<A>());
    for i in range(0, m).rev() {
        // b_ux_sum = b[i] - Sum(for j = i .. m) U_ij x_j
        let mut b_ux_sum = b[i].clone();
        for (uij, xj) in u.row_iter(i).rev().zip(x.iter().rev()).take((m - i - 1) as uint) {
            b_ux_sum = b_ux_sum - (*uij) * (*xj);
        }
        x.as_mut_slice()[i as uint] = b_ux_sum / u[(i, i)];
    }
    Array::from_vec(x)
}
