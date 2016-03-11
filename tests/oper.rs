extern crate ndarray;
extern crate num as libnum;

use ndarray::RcArray;
use ndarray::{arr0, rcarr1, rcarr2};
use ndarray::{
    OwnedArray,
    Ix,
};

use std::fmt;
use libnum::Float;

fn test_oper(op: &str, a: &[f32], b: &[f32], c: &[f32])
{
    let aa = rcarr1(a);
    let bb = rcarr1(b);
    let cc = rcarr1(c);
    test_oper_arr(op, aa.clone(), bb.clone(), cc.clone());
    let dim = (2, 2);
    let aa = aa.reshape(dim);
    let bb = bb.reshape(dim);
    let cc = cc.reshape(dim);
    test_oper_arr(op, aa.clone(), bb.clone(), cc.clone());
    let dim = (1, 2, 1, 2);
    let aa = aa.reshape(dim);
    let bb = bb.reshape(dim);
    let cc = cc.reshape(dim);
    test_oper_arr(op, aa.clone(), bb.clone(), cc.clone());
}

fn test_oper_arr<A: Float + fmt::Debug, D: ndarray::Dimension>
    (op: &str, mut aa: RcArray<A,D>, bb: RcArray<A, D>, cc: RcArray<A, D>)
{
    match op {
        "+" => {
            assert_eq!(&aa + &bb, cc);
            aa.iadd(&bb);
            assert_eq!(aa, cc);
        },
        "-" => {
            assert_eq!(&aa - &bb, cc);
            aa.isub(&bb);
            assert_eq!(aa, cc);
        },
        "*" => {
            assert_eq!(&aa * &bb, cc);
            aa.imul(&bb);
            assert_eq!(aa, cc);
        },
        "/" => {
            assert_eq!(&aa / &bb, cc);
            aa.idiv(&bb);
            assert_eq!(aa, cc);
        },
        "%" => {
            assert_eq!(&aa % &bb, cc);
            aa.irem(&bb);
            assert_eq!(aa, cc);
        },
        "neg" => {
            assert_eq!(-aa.clone(), cc);
            aa.ineg();
            assert_eq!(aa, cc);
        },
        _ => panic!()
    }
}

#[test]
fn operations()
{
    test_oper("+", &[1.0,2.0,3.0,4.0], &[0.0, 1.0, 2.0, 3.0], &[1.0,3.0,5.0,7.0]);
    test_oper("-", &[1.0,2.0,3.0,4.0], &[0.0, 1.0, 2.0, 3.0], &[1.0,1.0,1.0,1.0]);
    test_oper("*", &[1.0,2.0,3.0,4.0], &[0.0, 1.0, 2.0, 3.0], &[0.0,2.0,6.0,12.0]);
    test_oper("/", &[1.0,2.0,3.0,4.0], &[1.0, 1.0, 2.0, 3.0], &[1.0,2.0,3.0/2.0,4.0/3.0]);
    test_oper("%", &[1.0,2.0,3.0,4.0], &[1.0, 1.0, 2.0, 3.0], &[0.0,0.0,1.0,1.0]);
    test_oper("neg", &[1.0,2.0,3.0,4.0], &[1.0, 1.0, 2.0, 3.0], &[-1.0,-2.0,-3.0,-4.0]);
}

#[test]
fn scalar_operations()
{
    let a = arr0::<f32>(1.);
    let b = rcarr1::<f32>(&[1., 1.]);
    let c = rcarr2(&[[1., 1.], [1., 1.]]);

    {
        let mut x = a.clone();
        let mut y = arr0(0.);
        x.iadd_scalar(&1.);
        y.assign_scalar(&2.);
        assert_eq!(x, a + arr0(1.));
        assert_eq!(x, y);
    }

    {
        let mut x = b.clone();
        let mut y = rcarr1(&[0., 0.]);
        x.iadd_scalar(&1.);
        y.assign_scalar(&2.);
        assert_eq!(x, b + arr0(1.));
        assert_eq!(x, y);
    }

    {
        let mut x = c.clone();
        let mut y = RcArray::zeros((2, 2));
        x.iadd_scalar(&1.);
        y.assign_scalar(&2.);
        assert_eq!(x, c + arr0(1.));
        assert_eq!(x, y);
    }
}

fn assert_approx_eq<F: fmt::Debug + Float>(f: F, g: F, tol: F) -> bool {
    assert!((f - g).abs() <= tol, "{:?} approx== {:?} (tol={:?})",
            f, g, tol);
    true
}

#[test]
fn dot_product() {
    let a = OwnedArray::range(0., 69., 1.);
    let b = &a * 2. - 7.;
    let dot = 197846.;
    assert_approx_eq(a.dot(&b), dot, 1e-5);
    let a = a.map(|f| *f as f32);
    let b = b.map(|f| *f as f32);
    assert_approx_eq(a.dot(&b), dot as f32, 1e-5);
    let a = a.map(|f| *f as i32);
    let b = b.map(|f| *f as i32);
    assert_eq!(a.dot(&b), dot as i32);
}

fn range_mat(m: Ix, n: Ix) -> OwnedArray<f32, (Ix, Ix)> {
    OwnedArray::linspace(0., (m * n - 1) as f32, m * n).into_shape((m, n)).unwrap()
}

#[cfg(has_assign)]
#[test]
fn mat_mul() {
    let (m, n, k) = (8, 8, 8);
    let a = range_mat(m, n);
    let b = range_mat(n, k);
    let mut b = b / 4.;
    {
        let mut c = b.column_mut(0);
        c += 1.0;
    }
    let ab = a.mat_mul(&b);

    let mut af = OwnedArray::zeros_f(a.dim());
    let mut bf = OwnedArray::zeros_f(b.dim());
    af.assign(&a);
    bf.assign(&b);

    assert_eq!(ab, a.mat_mul(&bf));
    assert_eq!(ab, af.mat_mul(&b));
    assert_eq!(ab, af.mat_mul(&bf));

    let (m, n, k) = (10, 5, 11);
    let a = range_mat(m, n);
    let b = range_mat(n, k);
    let mut b = b / 4.;
    {
        let mut c = b.column_mut(0);
        c += 1.0;
    }
    let ab = a.mat_mul(&b);

    let mut af = OwnedArray::zeros_f(a.dim());
    let mut bf = OwnedArray::zeros_f(b.dim());
    af.assign(&a);
    bf.assign(&b);

    assert_eq!(ab, a.mat_mul(&bf));
    assert_eq!(ab, af.mat_mul(&b));
    assert_eq!(ab, af.mat_mul(&bf));
}
