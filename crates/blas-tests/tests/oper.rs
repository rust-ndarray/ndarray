extern crate approx;
extern crate blas_src;
extern crate defmac;
extern crate ndarray;
extern crate num_complex;
extern crate num_traits;

use ndarray::prelude::*;

use ndarray::linalg::general_mat_mul;
use ndarray::linalg::general_mat_vec_mul;
use ndarray::Order;
use ndarray::{Data, Ix, LinalgScalar};
use ndarray_gen::array_builder::ArrayBuilder;
use ndarray_gen::array_builder::ElementGenerator;

use approx::assert_relative_eq;
use defmac::defmac;
use itertools::iproduct;
use num_complex::Complex32;
use num_complex::Complex64;
use num_traits::Num;

#[test]
fn mat_vec_product_1d()
{
    let a = arr2(&[[1.], [2.]]);
    let b = arr1(&[1., 2.]);
    let ans = arr1(&[5.]);
    assert_eq!(a.t().dot(&b), ans);
}

#[test]
fn mat_vec_product_1d_broadcast()
{
    let a = arr2(&[[1.], [2.], [3.]]);
    let b = arr1(&[1.]);
    let b = b.broadcast(3).unwrap();
    let ans = arr1(&[6.]);
    assert_eq!(a.t().dot(&b), ans);
}

#[test]
fn mat_vec_product_1d_inverted_axis()
{
    let a = arr2(&[[1.], [2.], [3.]]);
    let mut b = arr1(&[1., 2., 3.]);
    b.invert_axis(Axis(0));

    let ans = arr1(&[3. + 4. + 3.]);
    assert_eq!(a.t().dot(&b), ans);
}

fn range_mat<A: Num + Copy>(m: Ix, n: Ix) -> Array2<A>
{
    ArrayBuilder::new((m, n)).build()
}

fn range_mat_complex(m: Ix, n: Ix) -> Array2<Complex32>
{
    ArrayBuilder::new((m, n)).build()
}

fn range_mat_complex64(m: Ix, n: Ix) -> Array2<Complex64>
{
    ArrayBuilder::new((m, n)).build()
}

fn range1_mat64(m: Ix) -> Array1<f64>
{
    ArrayBuilder::new(m).build()
}

fn range_i32(m: Ix, n: Ix) -> Array2<i32>
{
    ArrayBuilder::new((m, n)).build()
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
            *rr = (0..k).fold(A::zero(), move |s, x| s + *lhs.uget((i, x)) * *rhs.uget((x, j)));
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
    reference_mat_mul(
        lhs,
        &rhs.as_standard_layout()
            .into_shape_with_order((k, 1))
            .unwrap(),
    )
    .into_shape_with_order(m)
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
    reference_mat_mul(
        &lhs.as_standard_layout()
            .into_shape_with_order((1, m))
            .unwrap(),
        rhs,
    )
    .into_shape_with_order(n)
    .unwrap()
}

// Check that matrix multiplication of contiguous matrices returns a
// matrix with the same order
#[test]
fn mat_mul_order()
{
    let (m, n, k) = (50, 50, 50);
    let a = range_mat::<f32>(m, n);
    let b = range_mat::<f32>(n, k);
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
fn mat_mul_broadcast()
{
    let (m, n, k) = (16, 16, 16);
    let a = range_mat::<f32>(m, n);
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
fn mat_mul_rev()
{
    let (m, n, k) = (16, 16, 16);
    let a = range_mat::<f32>(m, n);
    let b = range_mat::<f32>(n, k);
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
fn mat_mut_zero_len()
{
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
    #[cfg(feature = "half")]
    mat_mul_zero_len!(range_mat::<half::f16>);
    #[cfg(feature = "half")]
    mat_mul_zero_len!(range_mat::<half::bf16>);
    mat_mul_zero_len!(range_mat::<f32>);
    mat_mul_zero_len!(range_mat::<f64>);
    mat_mul_zero_len!(range_i32);
}

#[test]
fn gen_mat_mul()
{
    let alpha = -2.3;
    let beta = 3.14;
    let sizes = vec![
        (4, 4, 4),
        (8, 8, 8),
        (8, 8, 1),
        (1, 10, 10),
        (10, 1, 10),
        (10, 10, 1),
        (1, 10, 1),
        (10, 1, 1),
        (1, 1, 10),
        (4, 17, 3),
        (17, 3, 22),
        (19, 18, 2),
        (15, 16, 17),
        (67, 50, 62),
    ];
    let strides = &[1, 2, -1, -2];
    let cf_order = [Order::C, Order::F];
    let generator = [ElementGenerator::Sequential, ElementGenerator::Checkerboard];

    // test different strides and memory orders
    for (&s1, &s2, &gen) in iproduct!(strides, strides, &generator) {
        for &(m, k, n) in &sizes {
            for (ord1, ord2, ord3) in iproduct!(cf_order, cf_order, cf_order) {
                println!("Case s1={}, s2={}, gen={:?}, orders={:?}, {:?}, {:?}", s1, s2, gen, ord1, ord2, ord3);
                let a = ArrayBuilder::new((m, k))
                    .memory_order(ord1)
                    .generator(gen)
                    .build()
                    * 0.5;
                let b = ArrayBuilder::new((k, n)).memory_order(ord2).build();
                let mut c = ArrayBuilder::new((m, n)).memory_order(ord3).build();

                let mut answer = c.clone();

                {
                    let av;
                    let bv;
                    let mut cv;

                    if s1 != 1 || s2 != 1 {
                        av = a.slice(s![..;s1, ..;s2]);
                        bv = b.slice(s![..;s2, ..;s2]);
                        cv = c.slice_mut(s![..;s1, ..;s2]);
                    } else {
                        // different stride cases for slicing versus not sliced (for axes of
                        // len=1); so test not sliced here.
                        av = a.view();
                        bv = b.view();
                        cv = c.view_mut();
                    }

                    let answer_part: Array<f64, _> = alpha * reference_mat_mul(&av, &bv) + beta * &cv;
                    answer.slice_mut(s![..;s1, ..;s2]).assign(&answer_part);

                    general_mat_mul(alpha, &av, &bv, beta, &mut cv);
                }
                assert_relative_eq!(c, answer, epsilon = 1e-12, max_relative = 1e-7);
            }
        }
    }
}

// Test y = A x where A is f-order
#[test]
fn gemm_64_1_f()
{
    let a = range_mat::<f64>(64, 64).reversed_axes();
    let (m, n) = a.dim();
    // m x n  times n x 1  == m x 1
    let x = range_mat::<f64>(n, 1);
    let mut y = range_mat::<f64>(m, 1);
    let answer = reference_mat_mul(&a, &x) + &y;
    general_mat_mul(1.0, &a, &x, 1.0, &mut y);
    assert_relative_eq!(y, answer, epsilon = 1e-12, max_relative = 1e-7);
}

#[test]
fn gemm_c64_1_f()
{
    let a = range_mat_complex64(64, 64).reversed_axes();
    let (m, n) = a.dim();
    // m x n  times n x 1  == m x 1
    let x = range_mat_complex64(n, 1);
    let mut y = range_mat_complex64(m, 1);
    let answer = reference_mat_mul(&a, &x) + &y;
    general_mat_mul(Complex64::new(1.0, 0.), &a, &x, Complex64::new(1.0, 0.), &mut y);
    assert_relative_eq!(
        y.mapv(|i| i.norm_sqr()),
        answer.mapv(|i| i.norm_sqr()),
        epsilon = 1e-12,
        max_relative = 1e-7
    );
}

#[test]
fn gemm_c32_1_f()
{
    let a = range_mat_complex(64, 64).reversed_axes();
    let (m, n) = a.dim();
    // m x n  times n x 1  == m x 1
    let x = range_mat_complex(n, 1);
    let mut y = range_mat_complex(m, 1);
    let answer = reference_mat_mul(&a, &x) + &y;
    general_mat_mul(Complex32::new(1.0, 0.), &a, &x, Complex32::new(1.0, 0.), &mut y);
    assert_relative_eq!(
        y.mapv(|i| i.norm_sqr()),
        answer.mapv(|i| i.norm_sqr()),
        epsilon = 1e-12,
        max_relative = 1e-7
    );
}

#[test]
fn gemm_c64_actually_complex()
{
    let mut a = range_mat_complex64(4, 4);
    a = a.map(|&i| if i.re > 8. { i.conj() } else { i });
    let mut b = range_mat_complex64(4, 6);
    b = b.map(|&i| if i.re > 4. { i.conj() } else { i });
    let mut y = range_mat_complex64(4, 6);
    let alpha = Complex64::new(0., 1.0);
    let beta = Complex64::new(1.0, 1.0);
    let answer = alpha * reference_mat_mul(&a, &b) + beta * &y;
    general_mat_mul(alpha.clone(), &a, &b, beta.clone(), &mut y);
    assert_relative_eq!(
        y.mapv(|i| i.norm_sqr()),
        answer.mapv(|i| i.norm_sqr()),
        epsilon = 1e-12,
        max_relative = 1e-7
    );
}

#[test]
fn gen_mat_vec_mul()
{
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
                for order in [Order::C, Order::F] {
                    let a = ArrayBuilder::new((m, k)).memory_order(order).build();
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
fn vec_mat_mul()
{
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
                for order in [Order::C, Order::F] {
                    let b = ArrayBuilder::new((m, n)).memory_order(order).build();
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
