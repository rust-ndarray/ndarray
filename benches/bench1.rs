#![feature(test)]
#![allow(unused_imports)]

extern crate test;
#[macro_use(s)]
extern crate ndarray;
#[cfg(feature = "rblas")]
extern crate rblas;
#[cfg(feature = "rblas")]
use rblas::matrix::Matrix;

use ndarray::{
    OwnedArray,
    Axis,
};
use ndarray::{arr0, arr1, arr2};

use test::black_box;

#[bench]
fn map(bench: &mut test::Bencher)
{
    let a = OwnedArray::linspace(0., 127., 128).into_shape((8, 16)).unwrap();
    bench.iter(|| {
        a.map(|&x| 2. * x)
    });
}

#[bench]
fn sum_1d_regular(bench: &mut test::Bencher)
{
    let a = OwnedArray::<i32, _>::zeros(64 * 64);
    let a = black_box(a);
    bench.iter(|| {
        let mut sum = 0;
        for &elt in a.iter() {
            sum += elt;
        }
        sum
    });
}

#[bench]
fn sum_1d_raw(bench: &mut test::Bencher)
{
    // this is autovectorized to death (= great performance)
    let a = OwnedArray::<i32, _>::zeros(64 * 64);
    let a = black_box(a);
    bench.iter(|| {
        let mut sum = 0;
        for &elt in a.raw_data() {
            sum += elt;
        }
        sum
    });
}

#[bench]
fn sum_2d_regular(bench: &mut test::Bencher)
{
    let a = OwnedArray::<i32, _>::zeros((64, 64));
    let a = black_box(a);
    bench.iter(|| {
        let mut sum = 0;
        for &elt in a.iter() {
            sum += elt;
        }
        sum
    });
}

#[bench]
fn sum_2d_by_row(bench: &mut test::Bencher)
{
    let a = OwnedArray::<i32, _>::zeros((64, 64));
    let a = black_box(a);
    bench.iter(|| {
        let mut sum = 0;
        for row in a.inner_iter() {
            for &elt in row {
                sum += elt;
            }
        }
        sum
    });
}

#[bench]
fn sum_2d_raw(bench: &mut test::Bencher)
{
    // this is autovectorized to death (= great performance)
    let a = OwnedArray::<i32, _>::zeros((64, 64));
    let a = black_box(a);
    bench.iter(|| {
        let mut sum = 0;
        for &elt in a.raw_data() {
            sum += elt;
        }
        sum
    });
}

#[bench]
fn sum_2d_cutout(bench: &mut test::Bencher)
{
    let a = OwnedArray::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    bench.iter(|| {
        let mut sum = 0;
        for &elt in a.iter() {
            sum += elt;
        }
        sum
    });
}

#[bench]
fn sum_2d_cutout_fold(bench: &mut test::Bencher)
{
    let a = OwnedArray::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    bench.iter(|| {
        a.fold(0, |acc, elt| acc + *elt)
    });
}

#[bench]
fn scalar_sum_2d_regular(bench: &mut test::Bencher)
{
    let a = OwnedArray::<i32, _>::zeros((64, 64));
    let a = black_box(a);
    bench.iter(|| {
        a.scalar_sum()
    });
}

#[bench]
fn scalar_sum_2d_cutout(bench: &mut test::Bencher)
{
    let a = OwnedArray::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    bench.iter(|| {
        a.scalar_sum()
    });
}

#[bench]
fn sum_2d_cutout_by_row(bench: &mut test::Bencher)
{
    let a = OwnedArray::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    bench.iter(|| {
        let mut sum = 0;
        for row in 0..a.shape()[0] {
            for &elt in a.row(row) {
                sum += elt;
            }
        }
        sum
    });
}

#[bench]
fn sum_2d_cutout_outer_iter(bench: &mut test::Bencher)
{
    let a = OwnedArray::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    bench.iter(|| {
        let mut sum = 0;
        for row in a.inner_iter() {
            for &elt in row {
                sum += elt;
            }
        }
        sum
    });
}

#[bench]
fn sum_2d_transpose_regular(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::<i32, _>::zeros((64, 64));
    a.swap_axes(0, 1);
    let a = black_box(a);
    bench.iter(|| {
        let mut sum = 0;
        for &elt in a.iter() {
            sum += elt;
        }
        sum
    });
}

#[bench]
fn sum_2d_transpose_by_row(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::<i32, _>::zeros((64, 64));
    a.swap_axes(0, 1);
    let a = black_box(a);
    bench.iter(|| {
        let mut sum = 0;
        for row in 0..a.shape()[0] {
            for &elt in a.row(row) {
                sum += elt;
            }
        }
        sum
    });
}

#[bench]
fn scalar_sum_2d_float(bench: &mut test::Bencher)
{
    let a = OwnedArray::<f32, _>::zeros((64, 64));
    let a = black_box(a.view());
    bench.iter(|| {
        a.scalar_sum()
    });
}

#[bench]
fn scalar_sum_2d_float_cutout(bench: &mut test::Bencher)
{
    let a = OwnedArray::<f32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    bench.iter(|| {
        a.scalar_sum()
    });
}

#[bench]
fn add_2d_regular(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::<i32, _>::zeros((64, 64));
    let b = OwnedArray::<i32, _>::zeros((64, 64));
    let bv = b.view();
    bench.iter(|| {
        a.iadd(&bv);
    });
}

#[bench]
fn add_2d_assign_ops(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::<i32, _>::zeros((64, 64));
    let b = OwnedArray::<i32, _>::zeros((64, 64));
    let bv = b.view();
    bench.iter(|| {
        let mut x = a.view_mut();
        x += &bv;
        black_box(x);
    });
}

#[bench]
fn add_2d_cutout(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::<i32, _>::zeros((66, 66));
    let mut acut = a.slice_mut(s![1..-1, 1..-1]);
    let b = OwnedArray::<i32, _>::zeros((64, 64));
    let bv = b.view();
    bench.iter(|| {
        acut.iadd(&bv);
    });
}

#[bench]
fn add_2d_broadcast_1_to_2(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::<i32, _>::zeros((64, 64));
    let b = OwnedArray::<i32, _>::zeros(64);
    let bv = b.view();
    bench.iter(|| {
        a.iadd(&bv);
    });
}

#[bench]
fn add_2d_broadcast_0_to_2(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::<i32, _>::zeros((64, 64));
    let b = OwnedArray::<i32, _>::zeros(());
    let bv = b.view();
    bench.iter(|| {
        a.iadd(&bv);
    });
}

// This is for comparison with add_2d_broadcast_0_to_2
#[bench]
fn add_2d_0_to_2_iadd_scalar(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::<i32, _>::zeros((64, 64));
    let n = black_box(0);
    bench.iter(|| {
        a.iadd_scalar(&n);
    });
}

#[bench]
fn add_2d_transposed(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::<i32, _>::zeros((64, 64));
    a.swap_axes(0, 1);
    let b = OwnedArray::<i32, _>::zeros((64, 64));
    let bv = b.view();
    bench.iter(|| {
        a.iadd(&bv);
    });
}

#[bench]
fn add_2d_f32_regular(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::<f32, _>::zeros((64, 64));
    let b = OwnedArray::<f32, _>::zeros((64, 64));
    let bv = b.view();
    bench.iter(|| {
        a.iadd(&bv);
    });
}

#[cfg(feature = "rblas")]
#[bench]
fn add_2d_f32_blas(bench: &mut test::Bencher)
{
    use rblas::Axpy;
    use rblas::attribute::Transpose;
    use ndarray::blas::AsBlas;
    let mut a = OwnedArray::<f32, _>::zeros((64, 64));
    let b = OwnedArray::<f32, _>::zeros((64, 64));
    let len = a.len();
    let mut av = a.view_mut().into_shape(len).unwrap();
    let bv = b.view().into_shape(len).unwrap();
    bench.iter(|| {
        f32::axpy(&1., &bv.bv(), &mut av.bvm());
    });
}

#[bench]
fn muladd_2d_f32_regular(bench: &mut test::Bencher)
{
    let mut av = OwnedArray::<f32, _>::zeros((64, 64));
    let bv = OwnedArray::<f32, _>::zeros((64, 64));
    let scalar = 3.1415926535;
    bench.iter(|| {
        av.zip_mut_with(&bv, |a, &b| *a += scalar * b);
    });
}

#[cfg(feature = "rblas")]
#[bench]
fn muladd_2d_f32_blas(bench: &mut test::Bencher)
{
    use rblas::Axpy;
    use rblas::attribute::Transpose;
    use ndarray::blas::AsBlas;
    let mut a = OwnedArray::<f32, _>::zeros((64, 64));
    let b = OwnedArray::<f32, _>::zeros((64, 64));
    let scalar = 3.1415926535;
    let len = a.len();
    let mut av = a.view_mut().into_shape(len).unwrap();
    let bv = b.view().into_shape(len).unwrap();
    bench.iter(|| {
        f32::axpy(&scalar, &bv.bv(), &mut av.bvm());
    });
}

#[bench]
fn assign_scalar_2d_corder(bench: &mut test::Bencher)
{
    let a = OwnedArray::zeros((64, 64));
    let mut a = black_box(a);
    let s = 3.;
    bench.iter(move || a.assign_scalar(&s))
}

#[bench]
fn assign_scalar_2d_cutout(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::zeros((66, 66));
    let a = a.slice_mut(s![1..-1, 1..-1]);
    let mut a = black_box(a);
    let s = 3.;
    bench.iter(move || a.assign_scalar(&s))
}

#[bench]
fn assign_scalar_2d_forder(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::zeros((64, 64));
    a.swap_axes(0, 1);
    let mut a = black_box(a);
    let s = 3.;
    bench.iter(move || a.assign_scalar(&s))
}

#[bench]
fn assign_zero_2d_corder(bench: &mut test::Bencher)
{
    let a = OwnedArray::zeros((64, 64));
    let mut a = black_box(a);
    bench.iter(|| a.assign_scalar(&0.))
}

#[bench]
fn assign_zero_2d_cutout(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::zeros((66, 66));
    let a = a.slice_mut(s![1..-1, 1..-1]);
    let mut a = black_box(a);
    bench.iter(|| a.assign_scalar(&0.))
}

#[bench]
fn assign_zero_2d_forder(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::zeros((64, 64));
    a.swap_axes(0, 1);
    let mut a = black_box(a);
    bench.iter(|| a.assign_scalar(&0.))
}

#[bench]
fn bench_iter_diag(bench: &mut test::Bencher)
{
    let a = OwnedArray::<f32, _>::zeros((1024, 1024));
    bench.iter(|| for elt in a.diag() { black_box(elt); })
}

#[bench]
fn bench_row_iter(bench: &mut test::Bencher)
{
    let a = OwnedArray::<f32, _>::zeros((1024, 1024));
    let it = a.row(17);
    bench.iter(|| for elt in it.clone() { black_box(elt); })
}

#[bench]
fn bench_col_iter(bench: &mut test::Bencher)
{
    let a = OwnedArray::<f32, _>::zeros((1024, 1024));
    let it = a.column(17);
    bench.iter(|| for elt in it.clone() { black_box(elt); })
}

macro_rules! mat_mul {
    ($modname:ident, $ty:ident, $(($name:ident, $m:expr, $n:expr, $k:expr))+) => {
        mod $modname {
            use test::{black_box, Bencher};
            use ndarray::OwnedArray;
            $(
            #[bench]
            fn $name(bench: &mut Bencher)
            {
                let a = OwnedArray::<$ty, _>::zeros(($m, $n));
                let b = OwnedArray::<$ty, _>::zeros(($n, $k));
                let a = black_box(a.view());
                let b = black_box(b.view());
                bench.iter(|| a.mat_mul(&b));
            }
            )+
        }
    }
}

mat_mul!{mat_mul_f32, f32,
    (m004, 4, 4, 4)
    (m007, 7, 7, 7)
    (m008, 8, 8, 8)
    (m012, 12, 12, 12)
    (m016, 16, 16, 16)
    (m032, 32, 32, 32)
    (m064, 64, 64, 64)
    (m127, 127, 127, 127)
    (mix16x4, 32, 4, 32)
    (mix32x2, 32, 2, 32)
}

mat_mul!{mat_mul_i32, i32,
    (m004, 4, 4, 4)
    (m007, 7, 7, 7)
    (m008, 8, 8, 8)
    (m012, 12, 12, 12)
    (m016, 16, 16, 16)
    (m032, 32, 32, 32)
    (m064, 64, 64, 64)
}

#[cfg(feature = "rblas")]
#[bench]
fn bench_mat_mul_rblas_64(bench: &mut test::Bencher)
{
    use rblas::Gemm;
    use rblas::attribute::Transpose;
    use ndarray::blas::AsBlas;

    let a = OwnedArray::<f32, _>::zeros((64, 64));
    let b = OwnedArray::<f32, _>::zeros((64, 64));
    let mut c = OwnedArray::<f32, _>::zeros((64, 64));
    bench.iter(|| {
        // C ← α AB + β C
        f32::gemm(&1.,
                  Transpose::NoTrans, &a.bv(),
                  Transpose::NoTrans, &b.bv(),
                  &1., &mut c.bvm());
    });
}

#[bench]
fn create_iter_4d(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::from_elem((4, 5, 3, 2), 1.0);
    a.swap_axes(0, 1);
    a.swap_axes(2, 1);
    let v = black_box(a.view());

    bench.iter(|| {
        v.into_iter()
    });
}

#[bench]
fn bench_to_owned_n(bench: &mut test::Bencher)
{
    let a = OwnedArray::<f32, _>::zeros((32, 32));
    bench.iter(|| a.to_owned());
}

#[bench]
fn bench_to_owned_t(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::<f32, _>::zeros((32, 32));
    a.swap_axes(0, 1);
    bench.iter(|| a.to_owned());
}

#[bench]
fn equality_i32(bench: &mut test::Bencher)
{
    let a = OwnedArray::<i32, _>::zeros((64, 64));
    let b = OwnedArray::<i32, _>::zeros((64, 64));
    bench.iter(|| a == b);
}

#[bench]
fn equality_f32(bench: &mut test::Bencher)
{
    let a = OwnedArray::<f32, _>::zeros((64, 64));
    let b = OwnedArray::<f32, _>::zeros((64, 64));
    bench.iter(|| a == b);
}

#[bench]
fn dot_f32_16(bench: &mut test::Bencher)
{
    let a = OwnedArray::<f32, _>::zeros(16);
    let b = OwnedArray::<f32, _>::zeros(16);
    bench.iter(|| a.dot(&b));
}

#[bench]
fn dot_f32_20(bench: &mut test::Bencher)
{
    let a = OwnedArray::<f32, _>::zeros(20);
    let b = OwnedArray::<f32, _>::zeros(20);
    bench.iter(|| a.dot(&b));
}

#[bench]
fn dot_f32_32(bench: &mut test::Bencher)
{
    let a = OwnedArray::<f32, _>::zeros(32);
    let b = OwnedArray::<f32, _>::zeros(32);
    bench.iter(|| a.dot(&b));
}

#[bench]
fn dot_f32_256(bench: &mut test::Bencher)
{
    let a = OwnedArray::<f32, _>::zeros(256);
    let b = OwnedArray::<f32, _>::zeros(256);
    bench.iter(|| a.dot(&b));
}

#[bench]
fn dot_f32_1024(bench: &mut test::Bencher)
{
    let av = OwnedArray::<f32, _>::zeros(1024);
    let bv = OwnedArray::<f32, _>::zeros(1024);
    bench.iter(|| {
        av.dot(&bv)
    });
}

#[bench]
fn dot_extended(bench: &mut test::Bencher) {
    let m = 10;
    let n = 33;
    let k = 10;
    let av = OwnedArray::<f32, _>::zeros((m, n));
    let bv = OwnedArray::<f32, _>::zeros((n, k));
    let mut res = OwnedArray::<f32, _>::zeros((m, k));
    // make a manual simple matrix multiply to test
    bench.iter(|| {
        for i in 0..m {
            for j in 0..k {
                unsafe {
                    *res.uget_mut((i, j)) = av.row(i).dot(&bv.column(j));
                }
            }
        }
    })
}

#[bench]
fn means(bench: &mut test::Bencher) {
    let a = OwnedArray::from_iter(0..100_000i64);
    let a = a.into_shape((100, 1000)).unwrap();
    bench.iter(|| a.mean(Axis(0)));
}
