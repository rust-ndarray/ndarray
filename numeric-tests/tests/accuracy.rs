extern crate approx;
extern crate rand_distr;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray_rand::{RandomExt, F32};
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

use ndarray::prelude::*;
use ndarray::{
    Data,
    LinalgScalar,
};
use ndarray::linalg::general_mat_mul;

use rand_distr::Normal;

use approx::{assert_abs_diff_eq, assert_relative_eq};

// simple, slow, correct (hopefully) mat mul
fn reference_mat_mul<A, S, S2>(lhs: &ArrayBase<S, Ix2>, rhs: &ArrayBase<S2, Ix2>)
    -> Array<A, Ix2>
    where A: LinalgScalar,
          S: Data<Elem=A>,
          S2: Data<Elem=A>,
{
    let ((m, k), (_, n)) = (lhs.dim(), rhs.dim());
    let mut res_elems = Vec::<A>::with_capacity(m * n);
    unsafe {
        res_elems.set_len(m * n);
    }

    let mut i = 0;
    let mut j = 0;
    for rr in &mut res_elems {
        unsafe {
            *rr = (0..k).fold(A::zero(),
                move |s, x| s + *lhs.uget((i, x)) * *rhs.uget((x, j)));
        }
        j += 1;
        if j == n {
            j = 0;
            i += 1;
        }
    }
    unsafe {
        ArrayBase::from_shape_vec_unchecked((m, n), res_elems)
    }
}

fn gen<D>(d: D) -> Array<f32, D>
    where D: Dimension,
{
    Array::random(d, F32(Normal::new(0., 1.).unwrap()))
}
fn gen_f64<D>(d: D) -> Array<f64, D>
    where D: Dimension,
{
    Array::random(d, Normal::new(0., 1.).unwrap())
}

#[test]
fn accurate_eye_f32() {
    for i in 0..20 {
        let eye = Array::eye(i);
        for j in 0..20 {
            let a = gen(Ix2(i, j));
            let a2 = eye.dot(&a);
            assert_abs_diff_eq!(a, a2, epsilon = 1e-6);
            let a3 = a.t().dot(&eye);
            assert_abs_diff_eq!(a.t(), a3, epsilon = 1e-6);
        }
    }
    // pick a few random sizes
    let mut rng = SmallRng::from_entropy();
    for _ in 0..10 {
        let i = rng.gen_range(15, 512);
        let j = rng.gen_range(15, 512);
        println!("Testing size {} by {}", i, j);
        let a = gen(Ix2(i, j));
        let eye = Array::eye(i);
        let a2 = eye.dot(&a);
        assert_abs_diff_eq!(a, a2, epsilon = 1e-6);
        let a3 = a.t().dot(&eye);
        assert_abs_diff_eq!(a.t(), a3, epsilon = 1e-6);
    }
}

#[test]
fn accurate_eye_f64() {
    let abs_tol = 1e-15;
    for i in 0..20 {
        let eye = Array::eye(i);
        for j in 0..20 {
            let a = gen_f64(Ix2(i, j));
            let a2 = eye.dot(&a);
            assert_abs_diff_eq!(a, a2, epsilon = abs_tol);
            let a3 = a.t().dot(&eye);
            assert_abs_diff_eq!(a.t(), a3, epsilon = abs_tol);
        }
    }
    // pick a few random sizes
    let mut rng = SmallRng::from_entropy();
    for _ in 0..10 {
        let i = rng.gen_range(15, 512);
        let j = rng.gen_range(15, 512);
        println!("Testing size {} by {}", i, j);
        let a = gen_f64(Ix2(i, j));
        let eye = Array::eye(i);
        let a2 = eye.dot(&a);
        assert_abs_diff_eq!(a, a2, epsilon = 1e-6);
        let a3 = a.t().dot(&eye);
        assert_abs_diff_eq!(a.t(), a3, epsilon = 1e-6);
    }
}

#[test]
fn accurate_mul_f32() {
    // pick a few random sizes
    let mut rng = SmallRng::from_entropy();
    for i in 0..20 {
        let m = rng.gen_range(15, 512);
        let k = rng.gen_range(15, 512);
        let n = rng.gen_range(15, 1560);
        let a = gen(Ix2(m, k));
        let b = gen(Ix2(n, k));
        let b = b.t();
        let (a, b) = if i > 10 {
            (a.slice(s![..;2, ..;2]),
             b.slice(s![..;2, ..;2]))
        } else { (a.view(), b) };

        println!("Testing size {} by {} by {}", a.shape()[0], a.shape()[1], b.shape()[1]);
        let c = a.dot(&b);
        let reference = reference_mat_mul(&a, &b);

        assert_relative_eq!(c, reference, epsilon = 1e-4, max_relative = 1e-3);
    }
}

#[test]
fn accurate_mul_f32_general() {
    // pick a few random sizes
    let mut rng = SmallRng::from_entropy();
    for i in 0..20 {
        let m = rng.gen_range(15, 512);
        let k = rng.gen_range(15, 512);
        let n = rng.gen_range(15, 1560);
        let a = gen(Ix2(m, k));
        let b = gen(Ix2(n, k));
        let mut c = gen(Ix2(m, n));
        let b = b.t();
        let (a, b, mut c) = if i > 10 {
            (a.slice(s![..;2, ..;2]),
             b.slice(s![..;2, ..;2]),
             c.slice_mut(s![..;2, ..;2]))
        } else { (a.view(), b, c.view_mut()) };

        println!("Testing size {} by {} by {}", a.shape()[0], a.shape()[1], b.shape()[1]);
        general_mat_mul(1., &a, &b, 0., &mut c);
        let reference = reference_mat_mul(&a, &b);

        assert_relative_eq!(c, reference, epsilon = 1e-4, max_relative = 1e-3);
    }
}

#[test]
fn accurate_mul_f64() {
    // pick a few random sizes
    let mut rng = SmallRng::from_entropy();
    for i in 0..20 {
        let m = rng.gen_range(15, 512);
        let k = rng.gen_range(15, 512);
        let n = rng.gen_range(15, 1560);
        let a = gen_f64(Ix2(m, k));
        let b = gen_f64(Ix2(n, k));
        let b = b.t();
        let (a, b) = if i > 10 {
            (a.slice(s![..;2, ..;2]),
             b.slice(s![..;2, ..;2]))
        } else { (a.view(), b) };

        println!("Testing size {} by {} by {}", a.shape()[0], a.shape()[1], b.shape()[1]);
        let c = a.dot(&b);
        let reference = reference_mat_mul(&a, &b);

        assert_relative_eq!(c, reference, epsilon = 1e-12, max_relative = 1e-7);
    }
}

#[test]
fn accurate_mul_f64_general() {
    // pick a few random sizes
    let mut rng = SmallRng::from_entropy();
    for i in 0..20 {
        let m = rng.gen_range(15, 512);
        let k = rng.gen_range(15, 512);
        let n = rng.gen_range(15, 1560);
        let a = gen_f64(Ix2(m, k));
        let b = gen_f64(Ix2(n, k));
        let mut c = gen_f64(Ix2(m, n));
        let b = b.t();
        let (a, b, mut c) = if i > 10 {
            (a.slice(s![..;2, ..;2]),
             b.slice(s![..;2, ..;2]),
             c.slice_mut(s![..;2, ..;2]))
        } else { (a.view(), b, c.view_mut()) };

        println!("Testing size {} by {} by {}", a.shape()[0], a.shape()[1], b.shape()[1]);
        general_mat_mul(1., &a, &b, 0., &mut c);
        let reference = reference_mat_mul(&a, &b);

        assert_relative_eq!(c, reference, epsilon = 1e-12, max_relative = 1e-7);
    }
}

#[test]
fn accurate_mul_with_column_f64() {
    // pick a few random sizes
    let mut rng = SmallRng::from_entropy();
    for i in 0..10 {
        let m = rng.gen_range(1, 350);
        let k = rng.gen_range(1, 350);
        let a = gen_f64(Ix2(m, k));
        let b_owner = gen_f64(Ix2(k, k));
        let b_row_col;
        let b_sq;

        // pick dense square or broadcasted to square matrix
        match i {
            0 ..= 3 => b_sq = b_owner.view(),
            4 ..= 7 => {
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
