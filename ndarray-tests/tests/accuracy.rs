
#[macro_use(s)]
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray_rand::{RandomExt, F32};
use rand::Rng;

use ndarray::{
    ArrayBase,
    OwnedArray,
    Dimension,
    Data,
    DataMut,
    Axis, RemoveAxis,
    LinalgScalar,
    Ix,
};

use rand::distributions::Normal;

// simple, slow, correct (hopefully) mat mul
fn reference_mat_mul<A, S, S2>(lhs: &ArrayBase<S, (Ix, Ix)>, rhs: &ArrayBase<S2, (Ix, Ix)>)
    -> OwnedArray<A, (Ix, Ix)>
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
        ArrayBase::from_vec_dim_unchecked((m, n), res_elems)
    }
}

fn gen<D>(d: D) -> OwnedArray<f32, D>
    where D: Dimension,
{
    OwnedArray::random(d, F32(Normal::new(0., 1.)))
}
fn gen_f64<D>(d: D) -> OwnedArray<f64, D>
    where D: Dimension,
{
    OwnedArray::random(d, Normal::new(0., 1.))
}

#[test]
fn accurate_eye_f32() {
    for i in 0..20 {
        let eye = OwnedArray::eye(i);
        for j in 0..20 {
            let a = gen((i, j));
            let a2 = eye.dot(&a);
            if !a.all_close(&a2, 1e-6) {
                panic!("Arrays are not equal:\n{:?}\n{:?}\n{:?}", a, a2, &a2 - &a);
            }
            let a3 = a.t().dot(&eye);
            if !a.t().all_close(&a3, 1e-6) {
                panic!("Arrays are not equal:\n{:?}\n{:?}\n{:?}", a.t(), a3, &a3 - &a.t());
            }
        }
    }
    // pick a few random sizes
    let mut rng = rand::weak_rng();
    for _ in 0..10 {
        let i = rng.gen_range(15, 512);
        let j = rng.gen_range(15, 512);
        println!("Testing size {} by {}", i, j);
        let a = gen((i, j));
        let eye = OwnedArray::eye(i);
        let a2 = eye.dot(&a);
        if !a.all_close(&a2, 1e-6) {
            panic!("Arrays are not equal:\n{:?}\n{:?}\n{:?}", a, a2, &a2 - &a);
        }
        let a3 = a.t().dot(&eye);
        if !a.t().all_close(&a3, 1e-6) {
            panic!("Arrays are not equal:\n{:?}\n{:?}\n{:?}", a.t(), a3, &a3 - &a.t());
        }
    }
}

#[test]
fn accurate_eye_f64() {
    let abs_tol = 1e-15;
    for i in 0..20 {
        let eye = OwnedArray::eye(i);
        for j in 0..20 {
            let a = gen_f64((i, j));
            let a2 = eye.dot(&a);
            if !a.all_close(&a2, abs_tol) {
                panic!("Arrays are not equal:\n{:?}\n{:?}\n{:?}", a, a2, &a2 - &a);
            }
            let a3 = a.t().dot(&eye);
            if !a.t().all_close(&a3, abs_tol) {
                panic!("Arrays are not equal:\n{:?}\n{:?}\n{:?}", a.t(), a3, &a3 - &a.t());
            }
        }
    }
    // pick a few random sizes
    let mut rng = rand::weak_rng();
    for _ in 0..10 {
        let i = rng.gen_range(15, 512);
        let j = rng.gen_range(15, 512);
        println!("Testing size {} by {}", i, j);
        let a = gen_f64((i, j));
        let eye = OwnedArray::eye(i);
        let a2 = eye.dot(&a);
        if !a.all_close(&a2, 1e-6) {
            panic!("Arrays are not equal:\n{:?}\n{:?}\n{:?}", a, a2, &a2 - &a);
        }
        let a3 = a.t().dot(&eye);
        if !a.t().all_close(&a3, 1e-6) {
            panic!("Arrays are not equal:\n{:?}\n{:?}\n{:?}", a.t(), a3, &a3 - &a.t());
        }
    }
}

#[test]
fn accurate_mul_f32() {
    // pick a few random sizes
    let mut rng = rand::weak_rng();
    for i in 0..20 {
        let m = rng.gen_range(15, 512);
        let k = rng.gen_range(15, 512);
        let n = rng.gen_range(15, 1560);
        let a = gen((m, k));
        let b = gen((n, k));
        let b = b.t();
        let (a, b) = if i > 10 {
            (a.slice(s![..;2, ..;2]),
             b.slice(s![..;2, ..;2]))
        } else { (a.view(), b) };

        println!("Testing size {} by {} by {}", a.shape()[0], a.shape()[1], b.shape()[1]);
        let c = a.dot(&b);
        let reference = reference_mat_mul(&a, &b);
        let diff = (&c - &reference).mapv_into(f32::abs);

        let rtol = 1e-3;
        let atol = 1e-4;
        let crtol = c.mapv(|x| x.abs() * rtol);
        let tol = crtol + atol;
        let tol_m_diff = &diff - &tol;
        let maxdiff = *tol_m_diff.max();
        println!("diff offset from tolerance level= {:.2e}", maxdiff);
        if maxdiff > 0. {
            panic!("results differ");
        }
    }
}

#[test]
fn accurate_mul_f64() {
    // pick a few random sizes
    let mut rng = rand::weak_rng();
    for i in 0..20 {
        let m = rng.gen_range(15, 512);
        let k = rng.gen_range(15, 512);
        let n = rng.gen_range(15, 1560);
        let a = gen_f64((m, k));
        let b = gen_f64((n, k));
        let b = b.t();
        let (a, b) = if i > 10 {
            (a.slice(s![..;2, ..;2]),
             b.slice(s![..;2, ..;2]))
        } else { (a.view(), b) };

        println!("Testing size {} by {} by {}", a.shape()[0], a.shape()[1], b.shape()[1]);
        let c = a.dot(&b);
        let reference = reference_mat_mul(&a, &b);
        let diff = (&c - &reference).mapv_into(f64::abs);

        let rtol = 1e-7;
        let atol = 1e-12;
        let crtol = c.mapv(|x| x.abs() * rtol);
        let tol = crtol + atol;
        let tol_m_diff = &diff - &tol;
        let maxdiff = *tol_m_diff.max();
        println!("diff offset from tolerance level= {:.2e}", maxdiff);
        if maxdiff > 0. {
            panic!("results differ");
        }
    }
}



trait Utils {
    type Elem;
    type Dim;
    type Data;
    fn max(&self) -> &Self::Elem
        where Self::Elem: PartialOrd;
    fn lift<F>(&mut self, F)
        where F: FnMut(Self::Elem) -> Self::Elem, Self::Elem: Copy,
              Self::Data: DataMut;
    fn fold_axis<F>(&self, axis: Axis, f: F) -> OwnedArray<Self::Elem, <Self::Dim as RemoveAxis>::Smaller>
        where Self::Dim: RemoveAxis,
              Self::Elem: Clone,
              F: FnMut(&Self::Elem, &Self::Elem) -> Self::Elem;
}

impl<A, S, D> Utils for ArrayBase<S, D>
    where S: Data<Elem=A>,
          D: Dimension,
{
    type Elem = A;
    type Dim = D;
    type Data = S;

    fn max(&self) -> &A
        where A: PartialOrd
    {
        let mut iter = self.iter();
        if let Some(mut max) = iter.next() {
            for elt in iter {
                if elt > max {
                    max = elt;
                }
            }
            max
        } else {
            panic!("empty");
        }
    }
    fn lift<F>(&mut self, mut f: F)
        where F: FnMut(A) -> A, A: Copy,
              S: DataMut<Elem=A>

    {
        for elt in self {
            *elt = f(*elt);
        }
    }

    fn fold_axis<F>(&self, axis: Axis, mut f: F) -> OwnedArray<A, D::Smaller>
        where D: RemoveAxis,
              F: FnMut(&A, &A) -> A,
              A: Clone,
    {
        let len = self.shape()[axis.axis()];
        let mut row = self.subview(axis, 0).to_owned();
        for i in 1..len {
            let r2 = self.subview(axis, i);
            for (a, b) in row.iter_mut().zip(r2) {
                *a = f(a, b);
            }
        }
        row
    }
}
