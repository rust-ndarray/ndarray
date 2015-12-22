#![feature(test)]
#![allow(unused_imports)]
#![cfg_attr(feature = "assign_ops", feature(augmented_assignments))]

extern crate test;
#[macro_use(s)]
extern crate ndarray;
#[cfg(feature = "rblas")]
extern crate rblas;
#[cfg(feature = "rblas")]
use rblas::matrix::Matrix;

use ndarray::{
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
fn small_iter_1d(bench: &mut test::Bencher)
{
    let a = arr1::<f32>(&[1., 2., 2.,
                         3., 4., 4.,
                         3., 4., 4.,
                         3., 4., 4.,
                         5., 6., 6.]);
    bench.iter(|| for &elt in a.iter() { black_box(elt); })
}

#[bench]
fn small_iter_1d_raw(bench: &mut test::Bencher)
{
    let a = arr1::<f32>(&[1., 2., 2.,
                         3., 4., 4.,
                         3., 4., 4.,
                         3., 4., 4.,
                         5., 6., 6.]);
    bench.iter(|| for &elt in a.raw_data().iter() { black_box(elt); })
}

#[bench]
fn small_iter_2d(bench: &mut test::Bencher)
{
    let a = arr2::<f32, _>(&[[1., 2., 2.],
                          [3., 4., 4.],
                          [3., 4., 4.],
                          [3., 4., 4.],
                          [5., 6., 6.]]);
    bench.iter(|| for &elt in a.iter() { black_box(elt); })
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
    let av = a.view().slice(s![1..-1, 1..-1]);
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
    let av = a.view().slice(s![1..-1, 1..-1]);
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
    let av = a.view().slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    bench.iter(|| {
        a.scalar_sum()
    });
}

#[bench]
fn sum_2d_cutout_by_row(bench: &mut test::Bencher)
{
    let a = OwnedArray::<i32, _>::zeros((66, 66));
    let av = a.view().slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    bench.iter(|| {
        let mut sum = 0;
        for row in 0..a.shape()[0] {
            for &elt in a.row_iter(row) {
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
    let av = a.view().slice(s![1..-1, 1..-1]);
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
            for &elt in a.row_iter(row) {
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
    let av = a.view().slice(s![1..-1, 1..-1]);
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
        let _x = black_box(a.view_mut() + bv);
    });
}

#[cfg(feature = "assign_ops")]
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
        let _x = black_box(acut.view_mut() + bv);
    });
}

#[bench]
fn add_2d_broadcast_1_to_2(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::<i32, _>::zeros((64, 64));
    let b = OwnedArray::<i32, _>::zeros(64);
    let bv = b.view();
    bench.iter(|| {
        let _x = black_box(a.view_mut() + bv);
    });
}

#[bench]
fn add_2d_broadcast_0_to_2(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::<i32, _>::zeros((64, 64));
    let b = OwnedArray::<i32, _>::zeros(());
    let bv = b.view();
    bench.iter(|| {
        let _x = black_box(a.view_mut() + bv);
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
    let a = OwnedArray::zeros((64, 64));
    let mut a = black_box(a);
    let s = 3.;
    bench.iter(|| a.assign_scalar(&s))
}

#[bench]
fn assign_scalar_2d_cutout(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::zeros((66, 66));
    let mut a = a.slice_mut(s![1..-1, 1..-1]);
    let mut a = black_box(a);
    let s = 3.;
    bench.iter(|| a.assign_scalar(&s))
}

#[bench]
fn assign_scalar_2d_transposed_large(bench: &mut test::Bencher)
{
    let mut a = OwnedArray::zeros((64, 64));
    a.swap_axes(0, 1);
    let mut a = black_box(a);
    let s = 3.;
    bench.iter(|| a.assign_scalar(&s))
}

#[bench]
fn assign_scalar_2d_raw_large(bench: &mut test::Bencher)
{
    let a = OwnedArray::zeros((64, 64));
    let mut a = black_box(a);
    let s = 3.;
    bench.iter(|| for elt in a.raw_data_mut() { *elt = s; });
}

#[bench]
fn bench_iter_diag(bench: &mut test::Bencher)
{
    let a = OwnedArray::<f32, _>::zeros((1024, 1024));
    bench.iter(|| for elt in a.diag_iter() { black_box(elt); })
}

#[bench]
fn bench_row_iter(bench: &mut test::Bencher)
{
    let a = OwnedArray::<f32, _>::zeros((1024, 1024));
    let it = a.row_iter(17);
    bench.iter(|| for elt in it.clone() { black_box(elt); })
}

#[bench]
fn bench_col_iter(bench: &mut test::Bencher)
{
    let a = OwnedArray::<f32, _>::zeros((1024, 1024));
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
    let a = zeros::<f32, _>((32, 32));
    bench.iter(|| a.to_owned());
}

#[bench]
fn bench_to_owned_t(bench: &mut test::Bencher)
{
    let mut a = zeros::<f32, _>((32, 32));
    a.swap_axes(0, 1);
    bench.iter(|| a.to_owned());
}

#[bench]
fn equality(bench: &mut test::Bencher)
{
    let a = OwnedArray::<i32, _>::zeros((64, 64));
    let b = OwnedArray::<i32, _>::zeros((64, 64));
    bench.iter(|| a == b);
}
