#![feature(test)]
#![allow(unused_imports)]

extern crate test;
extern crate ndarray;
#[cfg(feature = "rblas")]
extern crate rblas;
#[cfg(feature = "rblas")]
use rblas::matrix::Matrix;

use ndarray::{Array, S, Si,
    OwnedArray,
    zeros,
};
use ndarray::{arr0, arr1, arr2};

use test::black_box;

#[bench]
fn bench_std_add_shared(bench: &mut test::Bencher)
{
    let mut a = arr2::<f32, _>(&[[1., 2., 2.],
                              [3., 4., 4.],
                              [3., 4., 4.],
                              [3., 4., 4.],
                              [5., 6., 6.]]);
    let b = a.clone();
    bench.iter(|| a.iadd(&b));
}

#[bench]
fn bench_std_add_owned(bench: &mut test::Bencher)
{
    let a = arr2::<f32, _>(&[[1., 2., 2.],
                              [3., 4., 4.],
                              [3., 4., 4.],
                              [3., 4., 4.],
                              [5., 6., 6.]]);
    let mut a = a.to_owned();
    let b = a.clone();
    bench.iter(|| a.iadd(&b));
}

#[bench]
fn bench_std_iter_1d(bench: &mut test::Bencher)
{
    let a = arr1::<f32>(&[1., 2., 2.,
                         3., 4., 4.,
                         3., 4., 4.,
                         3., 4., 4.,
                         5., 6., 6.]);
    bench.iter(|| for &elt in a.iter() { black_box(elt); })
}

#[bench]
fn bench_std_iter_1d_raw(bench: &mut test::Bencher)
{
    let a = arr1::<f32>(&[1., 2., 2.,
                         3., 4., 4.,
                         3., 4., 4.,
                         3., 4., 4.,
                         5., 6., 6.]);
    bench.iter(|| for &elt in a.raw_data().iter() { black_box(elt); })
}

#[bench]
fn bench_std_iter_2d(bench: &mut test::Bencher)
{
    let a = arr2::<f32, _>(&[[1., 2., 2.],
                          [3., 4., 4.],
                          [3., 4., 4.],
                          [3., 4., 4.],
                          [5., 6., 6.]]);
    bench.iter(|| for &elt in a.iter() { black_box(elt); })
}

#[bench]
fn bench_std_iter_1d_large(bench: &mut test::Bencher)
{
    let a = Array::<i32, _>::zeros(64 * 64);
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
fn bench_std_iter_1d_raw_large(bench: &mut test::Bencher)
{
    // this is autovectorized to death (= great performance)
    let a = Array::<i32, _>::zeros(64 * 64);
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
fn bench_std_iter_2d_large(bench: &mut test::Bencher)
{
    let a = Array::<i32, _>::zeros((64, 64));
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
fn bench_std_iter_2d_raw_large(bench: &mut test::Bencher)
{
    // this is autovectorized to death (= great performance)
    let a = Array::<i32, _>::zeros((64, 64));
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
fn assign_scalar_2d_transposed(bench: &mut test::Bencher)
{
    let mut a = arr2::<f32, _>(&[[1., 2., 2.],
                              [3., 4., 4.],
                              [3., 4., 4.],
                              [3., 4., 4.],
                              [5., 6., 6.]]);
    a.swap_axes(0, 1);
    bench.iter(|| a.assign_scalar(&3.))
}

#[bench]
fn assign_scalar_2d(bench: &mut test::Bencher)
{
    let mut a = arr2::<f32, _>(&[[1., 2., 2.],
                              [3., 4., 4.],
                              [3., 4., 4.],
                              [3., 4., 4.],
                              [5., 6., 6.]]);
    bench.iter(|| a.assign_scalar(&3.))
}

#[bench]
fn assign_scalar_2d_large(bench: &mut test::Bencher)
{
    let a = Array::zeros((64, 64));
    let mut a = black_box(a);
    let s = black_box(3.);
    bench.iter(|| a.assign_scalar(&s))
}

#[bench]
fn assign_scalar_2d_transposed_large(bench: &mut test::Bencher)
{
    let mut a = Array::zeros((64, 64));
    a.swap_axes(0, 1);
    let mut a = black_box(a);
    let s = black_box(3.);
    bench.iter(|| a.assign_scalar(&s))
}

#[bench]
fn assign_scalar_2d_raw_large(bench: &mut test::Bencher)
{
    let a = Array::zeros((64, 64));
    let mut a = black_box(a);
    let s = black_box(3.);
    bench.iter(|| for elt in a.raw_data_mut() { *elt = s; });
}

#[bench]
fn bench_iter_diag(bench: &mut test::Bencher)
{
    let a = Array::<f32, _>::zeros((1024, 1024));
    bench.iter(|| for elt in a.diag_iter() { black_box(elt); })
}

#[bench]
fn bench_row_iter(bench: &mut test::Bencher)
{
    let a = Array::<f32, _>::zeros((1024, 1024));
    let it = a.row_iter(17);
    bench.iter(|| for elt in it.clone() { black_box(elt); })
}

#[bench]
fn bench_col_iter(bench: &mut test::Bencher)
{
    let a = Array::<f32, _>::zeros((1024, 1024));
    let it = a.col_iter(17);
    bench.iter(|| for elt in it.clone() { black_box(elt); })
}

#[bench]
fn bench_mat_mul(bench: &mut test::Bencher)
{
    let a = arr2::<f32, _>(&[[1., 2., 2.],
                          [3., 4., 4.],
                          [3., 4., 4.],
                          [3., 4., 4.],
                          [5., 6., 6.]]);
    let mut at = a.clone();
    at.swap_axes(0, 1);
    bench.iter(|| at.mat_mul(&a));
}

#[bench]
fn bench_mat_mul_large(bench: &mut test::Bencher)
{
    let a = OwnedArray::<f32, _>::zeros((64, 64));
    let b = OwnedArray::<f32, _>::zeros((64, 64));
    let a = black_box(a.view());
    let b = black_box(b.view());
    bench.iter(|| a.mat_mul(&b));
}

#[cfg(feature = "rblas")]
#[bench]
fn bench_mat_mul_rblas_large(bench: &mut test::Bencher)
{
    use rblas::Gemm;
    use rblas::attribute::Transpose;
    use ndarray::blas::AsBlas;

    let mut a = OwnedArray::<f32, _>::zeros((64, 64));
    let mut b = OwnedArray::<f32, _>::zeros((64, 64));
    let mut c = OwnedArray::<f32, _>::zeros((64, 64));
    bench.iter(|| {
        // C ← α AB + β C
        f32::gemm(&1.,
                  Transpose::NoTrans, &a.blas(),
                  Transpose::NoTrans, &b.blas(),
                  &1., &mut c.blas());
    });
}

#[bench]
fn bench_create_iter(bench: &mut test::Bencher)
{
    let a = arr2(&[[1., 2., 2.],
                   [3., 4., 4.],
                   [3., 4., 4.],
                   [3., 4., 4.],
                   [5., 6., 6.]]);
    let mut at = a.clone();
    at.swap_axes(0, 1);
    let v = black_box(at.view());

    bench.iter(|| {
        let v = black_box(v);
        v.into_iter()
    });
}

#[bench]
fn lst_squares(bench: &mut test::Bencher)
{
    let xs = arr2::<f32, _>(&[[ 2.,  3.],
                           [-2., -1.],
                           [ 1.,  5.],
                           [-1.,  2.]]);
    let b = arr1(&[1., -1., 2., 1.]);
    bench.iter(|| ndarray::linalg::least_squares(&xs, &b));
}

#[bench]
fn bench_to_owned_n(bench: &mut test::Bencher)
{
    let mut a = zeros::<f32, _>((32, 32));
    bench.iter(|| a.to_owned());
}

#[bench]
fn bench_to_owned_t(bench: &mut test::Bencher)
{
    let mut a = zeros::<f32, _>((32, 32));
    a.swap_axes(0, 1);
    bench.iter(|| a.to_owned());
}
