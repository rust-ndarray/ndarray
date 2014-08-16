#![allow(uppercase_variables)]

//! A few linear algebra operations on two-dimensional arrays.

use std::num::{zero, one};

use super::{Array, Dimension, Ix};

/// Column vector.
pub type Col<A> = Array<A, Ix>;
/// Rectangular matrix.
pub type Mat<A> = Array<A, (Ix, Ix)>;


/// Return the identity matrix of dimension *n*.
pub fn eye<A: Num + Clone>(n: Ix) -> Mat<A>
{
    let mut eye = Array::zeros((n, n));
    for i in range(0, n) {
        eye[(i, i)] = one::<A>();
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
pub fn least_squares<A: Float>(a: &Mat<A>, b: &Col<A>) -> Col<A>
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
    let (m, n) = a.dim();

    let mut aT = a.clone();
    aT.swap_axes(0, 1);

    let aT_a = aT.mat_mul(a);
    let mut L = cholesky(&aT_a);
    let rhs = aT.mat_mul(&b.reshape((m, 1))).reshape(n);

    // Solve L z = aT b
    let z = subst_fw(&L, &rhs);

    // Solve L.T x = z
    L.swap_axes(0, 1);
    let x_lstsq = subst_bw(&L, &z);
    x_lstsq
}

/// Factor *A = L L.T*.
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
pub fn cholesky<A: Float>(a: &Mat<A>) -> Mat<A>
{
    let z = zero::<A>();
    let (m, n) = a.dim();
    assert!(m == n);
    let mut L = Array::<A, _>::zeros((n, n));
    for i in range(0, m) {
        // Entries 0 .. i before the diagonal
        for j in range(0, i) {
            // L²_1,1
            // L_2,1 L_1,1  L²_2,1 + L²_2,2
            // L_3,1 L_1,1  L_3,1 L_2,1 + L_3,2 L_2,2  L²_3,1 + L²_3,2 + L²_3,3
            let mut lik_ljk_sum = z.clone();
            for k in range(0, j) {
                lik_ljk_sum = lik_ljk_sum + L[(i, k)] * L[(j, k)];
            }

            L[(i, j)] = (a[(i, j)] - lik_ljk_sum) / L[(j, j)];
        }
        // diagonal where i == j
        // L_j,j = Sqrt[A_j,j - Sum_k=1 to (j-1) L²_j,k ]
        let j = i;
        let mut ljk_sum = z.clone();
        for k in range(0, j) {
            let ljk = L[(j, k)];
            ljk_sum = ljk_sum + ljk * ljk;
        }
        L[(j, j)] = (a[(j, j)] - ljk_sum).sqrt();
    }
    L
}

/// Solve *L x = b* where *L* is a lower triangular matrix.
pub fn subst_fw<A: Num + Clone>(l: &Mat<A>, b: &Col<A>) -> Col<A>
{
    let (m, n) = l.dim();
    assert!(m == n);
    assert!(m == b.dim());
    let mut res = Array::zeros(m);
    for i in range(0, m) {
        let mut b_lx_sum = b[i].clone();
        for j in range(0, i) {
            b_lx_sum = b_lx_sum - l[(i, j)] * res[j];
        }
        res[i] = b_lx_sum / l[(i, i)];
    }
    res
}

/// Solve *U x = b* where *U* is an upper triangular matrix.
pub fn subst_bw<A: Num + Clone>(u: &Mat<A>, b: &Col<A>) -> Col<A>
{
    let (m, n) = u.dim();
    assert!(m == n);
    assert!(m == b.dim());
    let mut res = Array::zeros(m);
    for i in range(0, m).rev() {
        let mut b_lx_sum = b[i].clone();
        for j in range(i, m).rev() {
            b_lx_sum = b_lx_sum - u[(i, j)] * res[j];
        }
        res[i] = b_lx_sum / u[(i, i)];
    }
    res
}
