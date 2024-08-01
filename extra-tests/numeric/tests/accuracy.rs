extern crate approx;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate rand_distr;

extern crate numeric_tests;

use std::fmt;

use ndarray_rand::RandomExt;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use ndarray::linalg::general_mat_mul;
use ndarray::prelude::*;
use ndarray::{Data, LinalgScalar};

use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use rand_distr::{Distribution, Normal, StandardNormal};

use approx::{assert_abs_diff_eq, assert_relative_eq};

fn kahan_sum<A>(iter: impl Iterator<Item = A>) -> A
where A: LinalgScalar
{
    let mut sum = A::zero();
    let mut compensation = A::zero();

    for elt in iter {
        let y = elt - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    sum
}

// simple, slow, correct (hopefully) mat mul
fn reference_mat_mul<A, S, S2>(lhs: &ArrayBase<S, Ix2>, rhs: &ArrayBase<S2, Ix2>) -> Array<A, Ix2>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let ((m, k), (_, n)) = (lhs.dim(), rhs.dim());
    let mut res_elems = Array::zeros(m * n);

    let mut i = 0;
    let mut j = 0;
    for rr in &mut res_elems {
        let lhs_i = lhs.row(i);
        let rhs_j = rhs.column(j);
        *rr = kahan_sum((0..k).map(move |x| lhs_i[x] * rhs_j[x]));

        j += 1;
        if j == n {
            j = 0;
            i += 1;
        }
    }

    res_elems.into_shape_with_order((m, n)).unwrap()
}

fn gen<A, D>(d: D, rng: &mut SmallRng) -> Array<A, D>
where
    D: Dimension,
    A: Float,
    StandardNormal: Distribution<A>,
{
    Array::random_using(d, Normal::new(A::zero(), A::one()).unwrap(), rng)
}

fn gen_complex<A, D>(d: D, rng: &mut SmallRng) -> Array<Complex<A>, D>
where
    D: Dimension,
    A: Float,
    StandardNormal: Distribution<A>,
{
    gen(d.clone(), rng).mapv(Complex::from) + gen(d, rng).mapv(|x| Complex::new(A::zero(), x))
}

#[test]
fn accurate_eye_f32()
{
    let rng = &mut SmallRng::from_entropy();
    for i in 0..20 {
        let eye = Array::eye(i);
        for j in 0..20 {
            let a = gen::<f32, _>(Ix2(i, j), rng);
            let a2 = eye.dot(&a);
            assert_abs_diff_eq!(a, a2, epsilon = 1e-6);
            let a3 = a.t().dot(&eye);
            assert_abs_diff_eq!(a.t(), a3, epsilon = 1e-6);
        }
    }
    // pick a few random sizes
    for _ in 0..10 {
        let i = rng.gen_range(15..512);
        let j = rng.gen_range(15..512);
        println!("Testing size {} by {}", i, j);
        let a = gen::<f32, _>(Ix2(i, j), rng);
        let eye = Array::eye(i);
        let a2 = eye.dot(&a);
        assert_abs_diff_eq!(a, a2, epsilon = 1e-6);
        let a3 = a.t().dot(&eye);
        assert_abs_diff_eq!(a.t(), a3, epsilon = 1e-6);
    }
}

#[test]
fn accurate_eye_f64()
{
    let rng = &mut SmallRng::from_entropy();
    let abs_tol = 1e-15;
    for i in 0..20 {
        let eye = Array::eye(i);
        for j in 0..20 {
            let a = gen::<f64, _>(Ix2(i, j), rng);
            let a2 = eye.dot(&a);
            assert_abs_diff_eq!(a, a2, epsilon = abs_tol);
            let a3 = a.t().dot(&eye);
            assert_abs_diff_eq!(a.t(), a3, epsilon = abs_tol);
        }
    }
    // pick a few random sizes
    for _ in 0..10 {
        let i = rng.gen_range(15..512);
        let j = rng.gen_range(15..512);
        println!("Testing size {} by {}", i, j);
        let a = gen::<f64, _>(Ix2(i, j), rng);
        let eye = Array::eye(i);
        let a2 = eye.dot(&a);
        assert_abs_diff_eq!(a, a2, epsilon = 1e-6);
        let a3 = a.t().dot(&eye);
        assert_abs_diff_eq!(a.t(), a3, epsilon = 1e-6);
    }
}

#[test]
fn accurate_mul_f32_dot()
{
    accurate_mul_float_general::<f32>(1e-5, false);
}

#[test]
fn accurate_mul_f32_general()
{
    accurate_mul_float_general::<f32>(1e-5, true);
}

#[test]
fn accurate_mul_f64_dot()
{
    accurate_mul_float_general::<f64>(1e-14, false);
}

#[test]
fn accurate_mul_f64_general()
{
    accurate_mul_float_general::<f64>(1e-14, true);
}

/// Generate random sized matrices using the given generator function.
/// Compute gemm using either .dot() (if use_general is false) otherwise general_mat_mul.
/// Return tuple of actual result matrix and reference matrix, which should be equal.
fn random_matrix_mul<A>(
    rng: &mut SmallRng, use_stride: bool, use_general: bool, generator: fn(Ix2, &mut SmallRng) -> Array2<A>,
) -> (Array2<A>, Array2<A>)
where A: LinalgScalar
{
    let m = rng.gen_range(15..512);
    let k = rng.gen_range(15..512);
    let n = rng.gen_range(15..1560);
    let a = generator(Ix2(m, k), rng);
    let b = generator(Ix2(n, k), rng);
    let c = if use_general {
        Some(generator(Ix2(m, n), rng))
    } else {
        None
    };

    let b = b.t();
    let (a, b, mut c) = if use_stride {
        (a.slice(s![..;2, ..;2]), b.slice(s![..;2, ..;2]), c.map(|c_| c_.slice_move(s![..;2, ..;2])))
    } else {
        (a.view(), b, c)
    };

    println!("Testing size {} by {} by {}", a.shape()[0], a.shape()[1], b.shape()[1]);
    if let Some(c) = &mut c {
        general_mat_mul(A::one(), &a, &b, A::zero(), c);
    } else {
        c = Some(a.dot(&b));
    }
    let c = c.unwrap();
    let reference = reference_mat_mul(&a, &b);

    (c, reference)
}

fn accurate_mul_float_general<A>(limit: f64, use_general: bool)
where
    A: Float + Copy + 'static + AsPrimitive<f64>,
    StandardNormal: Distribution<A>,
    A: fmt::Debug,
{
    // pick a few random sizes
    let mut rng = SmallRng::from_entropy();
    for i in 0..20 {
        let (c, reference) = random_matrix_mul(&mut rng, i > 10, use_general, gen::<A, _>);

        let diff = &c - &reference;
        let max_diff = diff.iter().copied().fold(A::zero(), A::max);
        let max_elt = reference.iter().copied().fold(A::zero(), A::max);
        println!("Max elt diff={:?}, max={:?}, ratio={:.4e}", max_diff, max_elt, (max_diff/max_elt).as_());
        assert!((max_diff / max_elt).as_() < limit,
                "Expected relative norm diff < {:e}, found {:?} / {:?}", limit, max_diff, max_elt);
    }
}

#[test]
fn accurate_mul_complex32()
{
    accurate_mul_complex_general::<f32>(1e-5);
}

#[test]
fn accurate_mul_complex64()
{
    accurate_mul_complex_general::<f64>(1e-14);
}

fn accurate_mul_complex_general<A>(limit: f64)
where
    A: Float + Copy + 'static + AsPrimitive<f64>,
    StandardNormal: Distribution<A>,
    A: fmt::Debug,
{
    // pick a few random sizes
    let mut rng = SmallRng::from_entropy();
    for i in 0..20 {
        let (c, reference) = random_matrix_mul(&mut rng, i > 10, true, gen_complex::<A, _>);

        let diff = &c - &reference;
        let max_elt = |elt: &Complex<_>| A::max(A::abs(elt.re), A::abs(elt.im));
        let max_diff = diff.iter().map(max_elt).fold(A::zero(), A::max);
        let max_elt = reference.iter().map(max_elt).fold(A::zero(), A::max);
        println!("Max elt diff={:?}, max={:?}, ratio={:.4e}", max_diff, max_elt, (max_diff/max_elt).as_());
        assert!((max_diff / max_elt).as_() < limit,
                "Expected relative norm diff < {:e}, found {:?} / {:?}", limit, max_diff, max_elt);
    }
}

#[test]
fn accurate_mul_with_column_f64()
{
    // pick a few random sizes
    let rng = &mut SmallRng::from_entropy();
    for i in 0..10 {
        let m = rng.gen_range(1..350);
        let k = rng.gen_range(1..350);
        let a = gen::<f64, _>(Ix2(m, k), rng);
        let b_owner = gen::<f64, _>(Ix2(k, k), rng);
        let b_row_col;
        let b_sq;

        // pick dense square or broadcasted to square matrix
        match i {
            0..=3 => b_sq = b_owner.view(),
            4..=7 => {
                b_row_col = b_owner.column(0);
                b_sq = b_row_col.broadcast((k, k)).unwrap();
            }
            _otherwise => {
                b_row_col = b_owner.row(0);
                b_sq = b_row_col.broadcast((k, k)).unwrap();
            }
        };

        for j in 0..k {
            for &flip in &[true, false] {
                let j = j as isize;
                let b = if flip {
                    // one row in 2D
                    b_sq.slice(s![j..j + 1, ..]).reversed_axes()
                } else {
                    // one column in 2D
                    b_sq.slice(s![.., j..j + 1])
                };
                println!("Testing size ({} × {}) by ({} × {})", a.shape()[0], a.shape()[1], b.shape()[0], b.shape()[1]);
                println!("Strides ({:?}) by ({:?})", a.strides(), b.strides());
                let c = a.dot(&b);
                let reference = reference_mat_mul(&a, &b);

                assert_relative_eq!(c, reference, epsilon = 1e-12, max_relative = 1e-7);
            }
        }
    }
}
