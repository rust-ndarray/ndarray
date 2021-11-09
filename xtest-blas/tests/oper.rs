extern crate approx;
extern crate blas_src;
extern crate defmac;
extern crate ndarray;
extern crate num_complex;
extern crate num_traits;

use ndarray::prelude::*;

use ndarray::linalg::general_mat_mul;
use ndarray::linalg::general_mat_vec_mul;
use ndarray::{Data, Ix, LinalgScalar};

use approx::assert_relative_eq;
use defmac::defmac;
use num_complex::Complex32;
use num_complex::Complex64;

#[test]
fn mat_vec_product_1d() {
    let a = arr2(&[[1.], [2.]]);
    let b = arr1(&[1., 2.]);
    let ans = arr1(&[5.]);
    assert_eq!(a.t().dot(&b), ans);
}

#[test]
fn mat_vec_product_1d_broadcast() {
    let a = arr2(&[[1.], [2.], [3.]]);
    let b = arr1(&[1.]);
    let b = b.broadcast(3).unwrap();
    let ans = arr1(&[6.]);
    assert_eq!(a.t().dot(&b), ans);
}

#[test]
fn mat_vec_product_1d_inverted_axis() {
    let a = arr2(&[[1.], [2.], [3.]]);
    let mut b = arr1(&[1., 2., 3.]);
    b.invert_axis(Axis(0));

    let ans = arr1(&[3. + 4. + 3.]);
    assert_eq!(a.t().dot(&b), ans);
}

fn range_mat(m: Ix, n: Ix) -> Array2<f32> {
    Array::linspace(0., (m * n) as f32 - 1., m * n)
        .into_shape((m, n))
        .unwrap()
}

fn range_mat64(m: Ix, n: Ix) -> Array2<f64> {
    Array::linspace(0., (m * n) as f64 - 1., m * n)
        .into_shape((m, n))
        .unwrap()
}

fn range_mat_complex(m: Ix, n: Ix) -> Array2<Complex32> {
    Array::linspace(0., (m * n) as f32 - 1., m * n)
        .into_shape((m, n))
        .unwrap()
        .map(|&f| Complex32::new(f, 0.))
}

fn range_mat_complex64(m: Ix, n: Ix) -> Array2<Complex64> {
    Array::linspace(0., (m * n) as f64 - 1., m * n)
        .into_shape((m, n))
        .unwrap()
        .map(|&f| Complex64::new(f, 0.))
}

fn range1_mat64(m: Ix) -> Array1<f64> {
    Array::linspace(0., m as f64 - 1., m)
}

fn range_i32(m: Ix, n: Ix) -> Array2<i32> {
    Array::from_iter(0..(m * n) as i32)
        .into_shape((m, n))
        .unwrap()
}

// simple, slow, correct (hopefully) mat mul
fn reference_mat_mul<A, S, S2>(lhs: &ArrayBase<S, Ix2>, rhs: &ArrayBase<S2, Ix2>) -> Array2<A>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let ((m, k), (k2, n)) = (lhs.dim(), rhs.dim());
    assert!(m.checked_mul(n).is_some());
    assert_eq!(k, k2);
    let mut res_elems = Vec::<A>::with_capacity(m * n);
    unsafe {
        res_elems.set_len(m * n);
    }

    let mut i = 0;
    let mut j = 0;
    for rr in &mut res_elems {
        unsafe {
            *rr = (0..k).fold(A::zero(), move |s, x| {
                s + *lhs.uget((i, x)) * *rhs.uget((x, j))
            });
        }
        j += 1;
        if j == n {
            j = 0;
            i += 1;
        }
    }
    unsafe { ArrayBase::from_shape_vec_unchecked((m, n), res_elems) }
}

// simple, slow, correct (hopefully) mat mul
fn reference_mat_vec_mul<A, S, S2>(lhs: &ArrayBase<S, Ix2>, rhs: &ArrayBase<S2, Ix1>) -> Array1<A>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let ((m, _), k) = (lhs.dim(), rhs.dim());
    reference_mat_mul(lhs, &rhs.as_standard_layout().into_shape((k, 1)).unwrap())
        .into_shape(m)
        .unwrap()
}

// simple, slow, correct (hopefully) mat mul
fn reference_vec_mat_mul<A, S, S2>(lhs: &ArrayBase<S, Ix1>, rhs: &ArrayBase<S2, Ix2>) -> Array1<A>
where
    A: LinalgScalar,
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let (m, (_, n)) = (lhs.dim(), rhs.dim());
    reference_mat_mul(&lhs.as_standard_layout().into_shape((1, m)).unwrap(), rhs)
        .into_shape(n)
        .unwrap()
}

// Check that matrix multiplication of contiguous matrices returns a
// matrix with the same order
#[test]
fn mat_mul_order() {
    let (m, n, k) = (50, 50, 50);
    let a = range_mat(m, n);
    let b = range_mat(n, k);
    let mut af = Array::zeros(a.dim().f());
    let mut bf = Array::zeros(b.dim().f());
    af.assign(&a);
    bf.assign(&b);

    let cc = a.dot(&b);
    let ff = af.dot(&bf);

    assert_eq!(cc.strides()[1], 1);
    assert_eq!(ff.strides()[0], 1);
}

// Check that matrix multiplication
// supports broadcast arrays.
#[test]
fn mat_mul_broadcast() {
    let (m, n, k) = (16, 16, 16);
    let a = range_mat(m, n);
    let x1 = 1.;
    let x = Array::from(vec![x1]);
    let b0 = x.broadcast((n, k)).unwrap();
    let b1 = Array::from_elem(n, x1);
    let b1 = b1.broadcast((n, k)).unwrap();
    let b2 = Array::from_elem((n, k), x1);

    let c2 = a.dot(&b2);
    let c1 = a.dot(&b1);
    let c0 = a.dot(&b0);
    assert_eq!(c2, c1);
    assert_eq!(c2, c0);
}

// Check that matrix multiplication supports reversed axes
#[test]
fn mat_mul_rev() {
    let (m, n, k) = (16, 16, 16);
    let a = range_mat(m, n);
    let b = range_mat(n, k);
    let mut rev = Array::zeros(b.dim());
    let mut rev = rev.slice_mut(s![..;-1, ..]);
    rev.assign(&b);
    println!("{:.?}", rev);

    let c1 = a.dot(&b);
    let c2 = a.dot(&rev);
    assert_eq!(c1, c2);
}

// Check that matrix multiplication supports arrays with zero rows or columns
#[test]
fn mat_mut_zero_len() {
    defmac!(mat_mul_zero_len range_mat_fn => {
        for n in 0..4 {
            for m in 0..4 {
                let a = range_mat_fn(m, n);
                let b = range_mat_fn(n, 0);
                assert_eq!(a.dot(&b), Array2::zeros((m, 0)));
            }
            for k in 0..4 {
                let a = range_mat_fn(0, n);
                let b = range_mat_fn(n, k);
                assert_eq!(a.dot(&b), Array2::zeros((0, k)));
            }
        }
    });
    mat_mul_zero_len!(range_mat);
    mat_mul_zero_len!(range_mat64);
    mat_mul_zero_len!(range_i32);
}

#[test]
fn gen_mat_mul() {
    let alpha = -2.3;
    let beta = 3.14;
    let sizes = vec![
        (4, 4, 4),
        (8, 8, 8),
        (17, 15, 16),
        (4, 17, 3),
        (17, 3, 22),
        (19, 18, 2),
        (16, 17, 15),
        (15, 16, 17),
        (67, 63, 62),
    ];
    // test different strides
    for &s1 in &[1, 2, -1, -2] {
        for &s2 in &[1, 2, -1, -2] {
            for &(m, k, n) in &sizes {
                let a = range_mat64(m, k);
                let b = range_mat64(k, n);
                let mut c = range_mat64(m, n);
                let mut answer = c.clone();

                {
                    let a = a.slice(s![..;s1, ..;s2]);
                    let b = b.slice(s![..;s2, ..;s2]);
                    let mut cv = c.slice_mut(s![..;s1, ..;s2]);

                    let answer_part = alpha * reference_mat_mul(&a, &b) + beta * &cv;
                    answer.slice_mut(s![..;s1, ..;s2]).assign(&answer_part);

                    general_mat_mul(alpha, &a, &b, beta, &mut cv);
                }
                assert_relative_eq!(c, answer, epsilon = 1e-12, max_relative = 1e-7);
            }
        }
    }
}

// Test y = A x where A is f-order
#[test]
fn gemm_64_1_f() {
    let a = range_mat64(64, 64).reversed_axes();
    let (m, n) = a.dim();
    // m x n  times n x 1  == m x 1
    let x = range_mat64(n, 1);
    let mut y = range_mat64(m, 1);
    let answer = reference_mat_mul(&a, &x) + &y;
    general_mat_mul(1.0, &a, &x, 1.0, &mut y);
    assert_relative_eq!(y, answer, epsilon = 1e-12, max_relative = 1e-7);
}

#[test]
fn gemm_c64_1_f() {
    let a = range_mat_complex64(64, 64).reversed_axes();
    let (m, n) = a.dim();
    // m x n  times n x 1  == m x 1
    let x = range_mat_complex64(n, 1);
    let mut y = range_mat_complex64(m, 1);
    let answer = reference_mat_mul(&a, &x) + &y;
    general_mat_mul(
        Complex64::new(1.0, 0.),
        &a,
        &x,
        Complex64::new(1.0, 0.),
        &mut y,
    );
    assert_relative_eq!(
        y.mapv(|i| i.norm_sqr()),
        answer.mapv(|i| i.norm_sqr()),
        epsilon = 1e-12,
        max_relative = 1e-7
    );
}

#[test]
fn gemm_c32_1_f() {
    let a = range_mat_complex(64, 64).reversed_axes();
    let (m, n) = a.dim();
    // m x n  times n x 1  == m x 1
    let x = range_mat_complex(n, 1);
    let mut y = range_mat_complex(m, 1);
    let answer = reference_mat_mul(&a, &x) + &y;
    general_mat_mul(
        Complex32::new(1.0, 0.),
        &a,
        &x,
        Complex32::new(1.0, 0.),
        &mut y,
    );
    assert_relative_eq!(
        y.mapv(|i| i.norm_sqr()),
        answer.mapv(|i| i.norm_sqr()),
        epsilon = 1e-12,
        max_relative = 1e-7
    );
}

#[test]
fn gen_mat_vec_mul() {
    let alpha = -2.3;
    let beta = 3.14;
    let sizes = vec![
        (4, 4),
        (8, 8),
        (17, 15),
        (4, 17),
        (17, 3),
        (19, 18),
        (16, 17),
        (15, 16),
        (67, 63),
    ];
    // test different strides
    for &s1 in &[1, 2, -1, -2] {
        for &s2 in &[1, 2, -1, -2] {
            for &(m, k) in &sizes {
                for &rev in &[false, true] {
                    let mut a = range_mat64(m, k);
                    if rev {
                        a = a.reversed_axes();
                    }
                    let (m, k) = a.dim();
                    let b = range1_mat64(k);
                    let mut c = range1_mat64(m);
                    let mut answer = c.clone();

                    {
                        let a = a.slice(s![..;s1, ..;s2]);
                        let b = b.slice(s![..;s2]);
                        let mut cv = c.slice_mut(s![..;s1]);

                        let answer_part = alpha * reference_mat_vec_mul(&a, &b) + beta * &cv;
                        answer.slice_mut(s![..;s1]).assign(&answer_part);

                        general_mat_vec_mul(alpha, &a, &b, beta, &mut cv);
                    }
                    assert_relative_eq!(c, answer, epsilon = 1e-12, max_relative = 1e-7);
                }
            }
        }
    }
}

#[test]
fn vec_mat_mul() {
    let sizes = vec![
        (4, 4),
        (8, 8),
        (17, 15),
        (4, 17),
        (17, 3),
        (19, 18),
        (16, 17),
        (15, 16),
        (67, 63),
    ];
    // test different strides
    for &s1 in &[1, 2, -1, -2] {
        for &s2 in &[1, 2, -1, -2] {
            for &(m, n) in &sizes {
                for &rev in &[false, true] {
                    let mut b = range_mat64(m, n);
                    if rev {
                        b = b.reversed_axes();
                    }
                    let (m, n) = b.dim();
                    let a = range1_mat64(m);
                    let mut c = range1_mat64(n);
                    let mut answer = c.clone();

                    {
                        let b = b.slice(s![..;s1, ..;s2]);
                        let a = a.slice(s![..;s1]);

                        let answer_part = reference_vec_mat_mul(&a, &b);
                        answer.slice_mut(s![..;s2]).assign(&answer_part);

                        c.slice_mut(s![..;s2]).assign(&a.dot(&b));
                    }
                    assert_relative_eq!(c, answer, epsilon = 1e-12, max_relative = 1e-7);
                }
            }
        }
    }
}
