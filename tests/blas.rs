#![cfg(feature = "rblas")]

extern crate rblas;
#[macro_use] extern crate ndarray;

use rblas::Gemm;
use rblas::attribute::Transpose;

use ndarray::{
    OwnedArray,
};

use ndarray::blas::AsBlas;

#[test]
fn strided_matrix() {
    // smoke test, a matrix multiplication of uneven size
    let (n, m) = (45, 33);
    let mut a = OwnedArray::linspace(0., ((n * m) - 1) as f32, n as usize * m as usize ).into_shape((n, m)).unwrap();
    let mut b = OwnedArray::eye(m);
    let mut res = OwnedArray::zeros(a.dim());
    Gemm::gemm(&1., Transpose::NoTrans, &a.blas(), Transpose::NoTrans, &b.blas(),
               &0., &mut res.blas());
    assert_eq!(res, a);

    // matrix multiplication, strided
    let mut aprim = a.to_shared();
    aprim.islice(s![0..12, 0..11]);
    println!("{:?}", aprim.shape());
    println!("{:?}", aprim.strides());
    let mut b = OwnedArray::eye(aprim.shape()[1]);
    let mut res = OwnedArray::zeros(aprim.dim());
    Gemm::gemm(&1., Transpose::NoTrans, &aprim.blas(), Transpose::NoTrans, &b.blas(),
               &0., &mut res.blas());
    println!("{:?}", aprim);
    println!("{:?}", b);
    println!("{:?}", res);
    println!("{:?}", &res - &aprim);
    assert_eq!(res, aprim);

    // Transposed matrix multiply
    let (np, mp) = aprim.dim();
    let mut res = OwnedArray::zeros((mp, np));
    let mut b = OwnedArray::eye(np);
    Gemm::gemm(&1., Transpose::Trans, &aprim.blas(), Transpose::NoTrans, &b.blas(),
               &0., &mut res.blas());
    let mut at = aprim.clone();
    at.swap_axes(0, 1);
    assert_eq!(at, res);

    // strided, needs copy
    let mut abis = a.to_shared();
    abis.islice(s![0..12, ..;2]);
    println!("{:?}", abis.shape());
    println!("{:?}", abis.strides());
    let mut b = OwnedArray::eye(abis.shape()[1]);
    let mut res = OwnedArray::zeros(abis.dim());
    Gemm::gemm(&1., Transpose::NoTrans, &abis.blas(), Transpose::NoTrans, &b.blas(),
               &0., &mut res.blas());
    println!("{:?}", abis);
    println!("{:?}", b);
    println!("{:?}", res);
    println!("{:?}", &res - &abis);
    assert_eq!(res, abis);
}
