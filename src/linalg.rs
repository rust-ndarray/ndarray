#![allow(uppercase_variables)]

//! A few linear algebra operations on two-dimensional arrays.

use std::num::{zero, one};

use super::{Array, Dimension, Ix};

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

/// Return the inverse matrix of square matrix `a`.
pub fn inverse<A: Primitive>(a: &Mat<A>) -> Mat<A>
{
    fail!()
}

/// Solve `a x = b` for matrices
///
/// Using transpose: aT a x = aT b;  aT a being square gives
/// x_leastsq = inv(aT a) aT b
///
/// More efficient by Cholesky decomposition
///
pub fn least_squares<A: Primitive>(a: &Array<A, (Ix, Ix)>, b: &Array<A, Ix>) -> Array<A, Ix>
{
    fail!()
}

/// Factor A = L L*.
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
