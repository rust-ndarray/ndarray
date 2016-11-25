#![allow(non_snake_case)]

#[macro_use]
extern crate ndarray;
extern crate itertools;

use ndarray::{S, Si};
use ndarray::prelude::*;
use ndarray::{
    rcarr2,
    arr3,
};
use ndarray::indices;
use itertools::free::enumerate;

#[test]
fn test_matmul_rcarray()
{
    let mut A = RcArray::<usize, _>::zeros((2, 3));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let mut B = RcArray::<usize, _>::zeros((3, 4));
    for (i, elt) in B.iter_mut().enumerate() {
        *elt = i;
    }

    let c = A.dot(&B);
    println!("A = \n{:?}", A);
    println!("B = \n{:?}", B);
    println!("A x B = \n{:?}", c);
    unsafe {
        let result = RcArray::from_shape_vec_unchecked((2, 4), vec![20, 23, 26, 29, 56, 68, 80, 92]);
        assert_eq!(c.shape(), result.shape());
        assert!(c.iter().zip(result.iter()).all(|(a,b)| a == b));
        assert!(c == result);
    }
}

#[test]
fn test_mat_mul() {
    // smoke test, a big matrix multiplication of uneven size
    let (n, m) = (45, 33);
    let a = RcArray::linspace(0., ((n * m) - 1) as f32, n as usize * m as usize ).reshape((n, m));
    let b = RcArray::eye(m);
    assert_eq!(a.dot(&b), a);
    let c = RcArray::eye(n);
    assert_eq!(c.dot(&a), a);
}


#[test]
fn test_slice()
{
    let mut A = RcArray::<usize, _>::zeros((3, 4));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let vi = A.slice(s![1.., ..;2]);
    assert_eq!(vi.shape(), &[2, 2]);
    let vi = A.slice(&[S, S]);
    assert_eq!(vi.shape(), A.shape());
    assert!(vi.iter().zip(A.iter()).all(|(a, b)| a == b));
}

#[should_panic]
#[test]
fn index_out_of_bounds() {
    let mut a = Array::<i32, _>::zeros((3, 4));
    a[[3, 2]] = 1;
}

#[should_panic]
#[test]
fn slice_oob()
{
    let a = RcArray::<i32, _>::zeros((3, 4));
    let _vi = a.slice(&[Si(0, Some(10), 1), S]);
}

#[test]
fn test_index()
{
    let mut A = RcArray::<usize, _>::zeros((2, 3));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    for ((i, j), a) in indices((2, 3)).zip(A.iter()) {
        assert_eq!(*a, A[[i, j]]);
    }

    let vi = A.slice(&[Si(1, None, 1), Si(0, None, 2)]);
    let mut it = vi.iter();
    for ((i, j), x) in indices((1, 2)).zip(it.by_ref()) {
        assert_eq!(*x, vi[[i, j]]);
    }
    assert!(it.next().is_none());
}

#[test]
fn test_index_arrays() {
    let a = Array1::from_iter(0..12);
    assert_eq!(a[1], a[[1]]);
    let v = a.view().into_shape((3, 4)).unwrap();
    assert_eq!(a[1], v[[0, 1]]);
    let w = v.into_shape((2, 2, 3)).unwrap();
    assert_eq!(a[1], w[[0, 0, 1]]);
}

#[test]
fn test_add()
{
    let mut A = RcArray::<usize, _>::zeros((2, 2));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let B = A.clone();
    A = A + &B;
    assert_eq!(A[[0, 0]], 0);
    assert_eq!(A[[0, 1]], 2);
    assert_eq!(A[[1, 0]], 4);
    assert_eq!(A[[1, 1]], 6);
}

#[test]
fn test_multidim()
{
    let mut mat = RcArray::zeros(2*3*4*5*6).reshape((2,3,4,5,6));
    mat[(0,0,0,0,0)] = 22u8;
    {
        for (i, elt) in mat.iter_mut().enumerate() {
            *elt = i as u8;
        }
    }
    assert_eq!(mat.shape(), &[2,3,4,5,6]);
}


/*
array([[[ 7,  6],
        [ 5,  4],
        [ 3,  2],
        [ 1,  0]],

       [[15, 14],
        [13, 12],
        [11, 10],
        [ 9,  8]]])
*/
#[test]
fn test_negative_stride_rcarray()
{
    let mut mat = RcArray::zeros((2, 4, 2));
    mat[[0, 0, 0]] = 1.0f32;
    for (i, elt) in mat.iter_mut().enumerate() {
        *elt = i as f32;
    }

    {
        let vi = mat.slice(&[S, Si(0, None, -1), Si(0, None, -1)]);
        assert_eq!(vi.shape(), &[2, 4, 2]);
        // Test against sequential iterator
        let seq = [7f32,6., 5.,4.,3.,2.,1.,0.,15.,14.,13., 12.,11.,  10.,   9.,   8.];
        for (a, b) in vi.clone().iter().zip(seq.iter()) {
            assert_eq!(*a, *b);
        }
    }
    {
        let vi = mat.slice(s![.., ..;-5, ..]);
        let seq = [6., 7., 14., 15.];
        for (a, b) in vi.iter().zip(seq.iter()) {
            assert_eq!(*a, *b);
        }
    }
}

#[test]
fn test_cow()
{
    let mut mat = RcArray::zeros((2,2));
    mat[[0, 0]] = 1;
    let n = mat.clone();
    mat[[0, 1]] = 2;
    mat[[1, 0]] = 3;
    mat[[1, 1]] = 4;
    assert_eq!(mat[[0, 0]], 1);
    assert_eq!(mat[[0, 1]], 2);
    assert_eq!(n[[0, 0]], 1);
    assert_eq!(n[[0, 1]], 0);
    assert_eq!(n.get((0, 1)), Some(&0));
    let mut rev = mat.reshape(4);
    rev.islice(&[Si(0, None, -1)]);
    assert_eq!(rev[0], 4);
    assert_eq!(rev[1], 3);
    assert_eq!(rev[2], 2);
    assert_eq!(rev[3], 1);
    let before = rev.clone();
    // mutation
    rev[0] = 5;
    assert_eq!(rev[0], 5);
    assert_eq!(rev[1], 3);
    assert_eq!(rev[2], 2);
    assert_eq!(rev[3], 1);
    assert_eq!(before[0], 4);
    assert_eq!(before[1], 3);
    assert_eq!(before[2], 2);
    assert_eq!(before[3], 1);
}

#[test]
fn test_sub()
{
    let mat = RcArray::linspace(0., 15., 16).reshape((2, 4, 2));
    let s1 = mat.subview(Axis(0), 0);
    let s2 = mat.subview(Axis(0), 1);
    assert_eq!(s1.shape(), &[4, 2]);
    assert_eq!(s2.shape(), &[4, 2]);
    let n = RcArray::linspace(8., 15., 8).reshape((4,2));
    assert_eq!(n, s2);
    let m = RcArray::from_vec(vec![2., 3., 10., 11.]).reshape((2, 2));
    assert_eq!(m, mat.subview(Axis(1), 1));
}


#[test]
fn test_select(){
    // test for 2-d array
    let x = arr2(&[[0., 1.], [1.,0.],[1.,0.],[1.,0.],[1.,0.],[0., 1.],[0., 1.]]);
    let r = x.select(Axis(0),&[1,3,5]);
    let c = x.select(Axis(1),&[1]);
    let r_target = arr2(&[[1.,0.],[1.,0.],[0., 1.]]);
    let c_target = arr2(&[[1.,0.,0.,0.,0., 1., 1.]]);
    assert!(r.all_close(&r_target,1e-8));
    assert!(c.all_close(&c_target.t(),1e-8));

    // test for 3-d array
    let y = arr3(&[[[1., 2., 3.],
                    [1.5, 1.5, 3.]],
                    [[1., 2., 8.],
                    [1., 2.5, 3.]]]);
    let r = y.select(Axis(1),&[1]);
    let c = y.select(Axis(2),&[1]);
    let r_target = arr3(&[[[1.5, 1.5, 3.]], [[1., 2.5, 3.]]]);
    let c_target = arr3(&[[[2.],[1.5]],[[2.],[2.5]]]);
    assert!(r.all_close(&r_target,1e-8));
    assert!(c.all_close(&c_target,1e-8));

}


#[test]
fn diag()
{
    let d = arr2(&[[1., 2., 3.0f32]]).into_diag();
    assert_eq!(d.dim(), 1);
    let a = arr2(&[[1., 2., 3.0f32], [0., 0., 0.]]);
    let d = a.view().into_diag();
    assert_eq!(d.dim(), 2);
    let d = arr2::<f32, _>(&[[]]).into_diag();
    assert_eq!(d.dim(), 0);
    let d = RcArray::<f32, _>::zeros(()).into_diag();
    assert_eq!(d.dim(), 1);
}

#[test]
fn swapaxes()
{
    let mut a = arr2(&[[1., 2.], [3., 4.0f32]]);
    let     b = arr2(&[[1., 3.], [2., 4.0f32]]);
    assert!(a != b);
    a.swap_axes(0, 1);
    assert_eq!(a, b);
    a.swap_axes(1, 1);
    assert_eq!(a, b);
    assert_eq!(a.as_slice_memory_order(), Some(&[1., 2., 3., 4.][..]));
    assert_eq!(b.as_slice_memory_order(), Some(&[1., 3., 2., 4.][..]));
}

#[test]
fn standard_layout()
{
    let mut a = arr2(&[[1., 2.], [3., 4.0]]);
    assert!(a.is_standard_layout());
    a.swap_axes(0, 1);
    assert!(!a.is_standard_layout());
    a.swap_axes(0, 1);
    assert!(a.is_standard_layout());
    let x1 = a.subview(Axis(0), 0);
    assert!(x1.is_standard_layout());
    let x2 = a.subview(Axis(1), 0);
    assert!(!x2.is_standard_layout());
}

#[test]
fn assign()
{
    let mut a = arr2(&[[1., 2.], [3., 4.]]);
    let     b = arr2(&[[1., 3.], [2., 4.]]);
    a.assign(&b);
    assert_eq!(a, b);

    /* Test broadcasting */
    a.assign(&RcArray::zeros(1));
    assert_eq!(a, RcArray::zeros((2, 2)));

    /* Test other type */
    a.assign(&Array::from_elem((2, 2), 3.));
    assert_eq!(a, RcArray::from_elem((2, 2), 3.));

    /* Test mut view */
    let mut a = arr2(&[[1, 2], [3, 4]]);
    {
        let mut v = a.view_mut();
        v.islice(&[Si(0, Some(1), 1), S]);
        v.fill(0);
    }
    assert_eq!(a, arr2(&[[0, 0], [3, 4]]));
}

#[test]
fn sum_mean()
{
    let a = arr2(&[[1., 2.], [3., 4.]]);
    assert_eq!(a.sum(Axis(0)), arr1(&[4., 6.]));
    assert_eq!(a.sum(Axis(1)), arr1(&[3., 7.]));
    assert_eq!(a.mean(Axis(0)), arr1(&[2., 3.]));
    assert_eq!(a.mean(Axis(1)), arr1(&[1.5, 3.5]));
    assert_eq!(a.sum(Axis(1)).sum(Axis(0)), arr0(10.));
    assert_eq!(a.view().mean(Axis(1)), aview1(&[1.5, 3.5]));
    assert_eq!(a.scalar_sum(), 10.);
}

#[test]
fn iter_size_hint()
{
    let mut a = arr2(&[[1., 2.], [3., 4.]]);
    {
        let mut it = a.iter();
        assert_eq!(it.size_hint(), (4, Some(4)));
        it.next();
        assert_eq!(it.size_hint().0, 3);
        it.next();
        assert_eq!(it.size_hint().0, 2);
        it.next();
        assert_eq!(it.size_hint().0, 1);
        it.next();
        assert_eq!(it.size_hint().0, 0);
        assert!(it.next().is_none());
        assert_eq!(it.size_hint().0, 0);
    }

    a.swap_axes(0, 1);
    {
        let mut it = a.iter();
        assert_eq!(it.size_hint(), (4, Some(4)));
        it.next();
        assert_eq!(it.size_hint().0, 3);
        it.next();
        assert_eq!(it.size_hint().0, 2);
        it.next();
        assert_eq!(it.size_hint().0, 1);
        it.next();
        assert_eq!(it.size_hint().0, 0);
        assert!(it.next().is_none());
        assert_eq!(it.size_hint().0, 0);
    }
}

#[test]
fn zero_axes()
{
    let mut a = arr1::<f32>(&[]);
    for _ in a.iter() {
        assert!(false);
    }
    a.map(|_| assert!(false));
    a.map_inplace(|_| assert!(false));
    a.visit(|_| assert!(false));
    println!("{:?}", a);
    let b = arr2::<f32, _>(&[[], [], [], []]);
    println!("{:?}\n{:?}", b.shape(), b);

    // we can even get a subarray of b
    let bsub = b.subview(Axis(0), 2);
    assert_eq!(bsub.dim(), 0);
}

#[test]
fn equality()
{
    let a = arr2(&[[1., 2.], [3., 4.]]);
    let mut b = arr2(&[[1., 2.], [2., 4.]]);
    assert!(a != b);
    b[(1, 0)] = 3.;
    assert!(a == b);

    // make sure we can compare different shapes without failure.
    let c = arr2(&[[1., 2.]]);
    assert!(a != c);
}

#[test]
fn map1()
{
    let a = arr2(&[[1., 2.], [3., 4.]]);
    let b = a.map(|&x| (x / 3.) as isize);
    assert_eq!(b, arr2(&[[0, 0], [1, 1]]));
    // test map to reference with array's lifetime.
    let c = a.map(|x| x);
    assert_eq!(a[(0, 0)], *c[(0, 0)]);
}

#[test]
fn as_slice_memory_order()
{
    // test that mutation breaks sharing
    let a = rcarr2(&[[1., 2.], [3., 4.0f32]]);
    let mut b = a.clone();
    for elt in b.as_slice_memory_order_mut().unwrap() {
        *elt = 0.;
    }
    assert!(a != b, "{:?} != {:?}", a, b);
}

#[test]
fn owned_array1() {
    let mut a = Array::from_vec(vec![1, 2, 3, 4]);
    for elt in a.iter_mut() {
        *elt = 2;
    }
    for elt in a.iter() {
        assert_eq!(*elt, 2);
    }
    assert_eq!(a.shape(), &[4]);

    let mut a = Array::zeros((2, 2));
    let mut b = RcArray::zeros((2, 2));
    a[(1, 1)] = 3;
    b[(1, 1)] = 3;
    assert_eq!(a, b);

    let c = a.clone();

    let d1 = &a + &b;
    let d2 = a + b;
    assert!(c != d1);
    assert_eq!(d1, d2);
}

#[test]
fn owned_array_with_stride() {
    let v: Vec<_> = (0..12).collect();
    let dim = (2, 3, 2);
    let strides = (1, 4, 2);

    let a = Array::from_shape_vec(dim.strides(strides), v).unwrap();
    assert_eq!(a.strides(), &[1, 4, 2]);
}

macro_rules! assert_matches {
    ($value:expr, $pat:pat) => {
        match $value {
            $pat => {}
            ref err => panic!("assertion failed: `{}` matches `{}` found: {:?}",
                               stringify!($value), stringify!($pat), err),
        }
    }
}

#[test]
fn from_vec_dim_stride_empty_1d() {
    let empty: [f32; 0] = [];
    assert_matches!(Array::from_shape_vec(0.strides(1), empty.to_vec()),
                    Ok(_));
}

#[test]
fn from_vec_dim_stride_0d() {
    let empty: [f32; 0] = [];
    let one = [1.];
    let two = [1., 2.];
    // too few elements
    assert_matches!(Array::from_shape_vec(().strides(()), empty.to_vec()), Err(_));
    // exact number of elements
    assert_matches!(Array::from_shape_vec(().strides(()), one.to_vec()), Ok(_));
    // too many are ok
    assert_matches!(Array::from_shape_vec(().strides(()), two.to_vec()), Ok(_));
}

#[test]
fn from_vec_dim_stride_2d_1() {
    let two = [1., 2.];
    let d = Ix2(2, 1);
    let s = d.default_strides();
    assert_matches!(Array::from_shape_vec(d.strides(s), two.to_vec()), Ok(_));
}

#[test]
fn from_vec_dim_stride_2d_2() {
    let two = [1., 2.];
    let d = Ix2(1, 2);
    let s = d.default_strides();
    assert_matches!(Array::from_shape_vec(d.strides(s), two.to_vec()), Ok(_));
}

#[test]
fn from_vec_dim_stride_2d_3() {
    let a = arr3(&[[[1]],
                   [[2]],
                   [[3]]]);
    let d = a.raw_dim();
    let s = d.default_strides();
    assert_matches!(Array::from_shape_vec(d.strides(s), a.as_slice().unwrap().to_vec()), Ok(_));
}

#[test]
fn from_vec_dim_stride_2d_4() {
    let a = arr3(&[[[1]],
                   [[2]],
                   [[3]]]);
    let d = a.raw_dim();
    let s = d.fortran_strides();
    assert_matches!(Array::from_shape_vec(d.strides(s), a.as_slice().unwrap().to_vec()), Ok(_));
}

#[test]
fn from_vec_dim_stride_2d_5() {
    let a = arr3(&[[[1, 2, 3]]]);
    let d = a.raw_dim();
    let s = d.fortran_strides();
    assert_matches!(Array::from_shape_vec(d.strides(s), a.as_slice().unwrap().to_vec()), Ok(_));
}

#[test]
fn from_vec_dim_stride_2d_6() {
    let a = [1., 2., 3., 4., 5., 6.];
    let d = (2, 1, 1);
    let s = (2, 2, 1);
    assert_matches!(Array::from_shape_vec(d.strides(s), a.to_vec()), Ok(_));

    let d = (1, 2, 1);
    let s = (2, 2, 1);
    assert_matches!(Array::from_shape_vec(d.strides(s), a.to_vec()), Ok(_));
}

#[test]
fn from_vec_dim_stride_2d_7() {
    // empty arrays can have 0 strides
    let a: [f32; 0] = [];
    // [[]] shape=[4, 0], strides=[0, 1]
    let d = (4, 0);
    let s = (0, 1);
    assert_matches!(Array::from_shape_vec(d.strides(s), a.to_vec()), Ok(_));
}

#[test]
fn from_vec_dim_stride_2d_8() {
    // strides must be strictly positive (nonzero)
    let a = [1.];
    let d = (1, 1);
    let s = (0, 1);
    assert_matches!(Array::from_shape_vec(d.strides(s), a.to_vec()), Err(_));
}

#[test]
fn from_vec_dim_stride_2d_rejects() {
    let two = [1., 2.];
    let d = (2, 2);
    let s = (1, 0);
    assert_matches!(Array::from_shape_vec(d.strides(s), two.to_vec()), Err(_));

    let d = (2, 2);
    let s = (0, 1);
    assert_matches!(Array::from_shape_vec(d.strides(s), two.to_vec()), Err(_));
}

#[test]
fn views() {
    let a = RcArray::from_vec(vec![1, 2, 3, 4]).reshape((2, 2));
    let b = a.view();
    assert_eq!(a, b);
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.clone() + a.clone(), &b + &b);
    assert_eq!(a.clone() + b, &b + &b);
    a.clone()[(0, 0)] = 99;
    assert_eq!(b[(0, 0)], 1);

    assert_eq!(a.view().into_iter().cloned().collect::<Vec<_>>(),
               vec![1, 2, 3, 4]);
}

#[test]
fn view_mut() {
    let mut a = RcArray::from_vec(vec![1, 2, 3, 4]).reshape((2, 2));
    for elt in &mut a.view_mut() {
        *elt = 0;
    }
    assert_eq!(a, Array::zeros((2, 2)));
    {
        let mut b = a.view_mut();
        b[(0, 0)] = 7;
    }
    assert_eq!(a[(0, 0)], 7);

    for elt in a.view_mut() {
        *elt = 2;
    }
    assert_eq!(a, RcArray::from_elem((2, 2), 2));
}

#[test]
fn slice_mut() {
    let mut a = RcArray::from_vec(vec![1, 2, 3, 4]).reshape((2, 2));
    for elt in a.slice_mut(&[S, S]) {
        *elt = 0;
    }
    assert_eq!(a, aview2(&[[0, 0], [0, 0]]));

    let mut b = arr2(&[[1, 2, 3],
                       [4, 5, 6]]);
    let c = b.clone(); // make sure we can mutate b even if it has to be unshared first
    for elt in b.slice_mut(&[S, Si(0, Some(1), 1)]) {
        *elt = 0;
    }
    assert_eq!(b, aview2(&[[0, 2, 3],
                           [0, 5, 6]]));
    assert!(c != b);

    for elt in b.slice_mut(&[S, Si(0, None, 2)]) {
        *elt = 99;
    }
    assert_eq!(b, aview2(&[[99, 2, 99],
                           [99, 5, 99]]));
}

#[test]
fn assign_ops()
{
    let mut a = arr2(&[[1., 2.], [3., 4.]]);
    let     b = arr2(&[[1., 3.], [2., 4.]]);
    (*&mut a.view_mut()) += &b;
    assert_eq!(a, arr2(&[[2., 5.], [5., 8.]]));

    a -= &b;
    a -= &b;
    assert_eq!(a, arr2(&[[0., -1.,], [1., 0.]]));

    a += 1.;
    assert_eq!(a, arr2(&[[1., 0.,], [2., 1.]]));
    a *= 10.;
    a /= 5.;
    assert_eq!(a, arr2(&[[2., 0.,], [4., 2.]]));
}

#[test]
fn aview() {
    let a = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
    let data = [[1., 2., 3.], [4., 5., 6.]];
    let b = aview2(&data);
    assert_eq!(a, b);
    assert_eq!(b.shape(), &[2, 3]);
}

#[test]
fn aview_mut() {
    let mut data = [0; 16];
    {
        let mut a = aview_mut1(&mut data).into_shape((4, 4)).unwrap();
        {
            let mut slc = a.slice_mut(s![..2, ..;2]);
            slc += 1;
        }
    }
    assert_eq!(data, [1, 0, 1, 0,  1, 0, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0]);
}

#[test]
fn transpose_view() {
    let a = arr2(&[[1, 2],
                   [3, 4]]);
    let at = a.view().reversed_axes();
    assert_eq!(at, arr2(&[[1, 3], [2, 4]]));

    let a = arr2(&[[1, 2, 3],
                   [4, 5, 6]]);
    let at = a.view().reversed_axes();
    assert_eq!(at, arr2(&[[1, 4], [2, 5], [3, 6]]));
}

#[test]
fn transpose_view_mut() {
    let mut a = arr2(&[[1, 2],
                       [3, 4]]);
    let mut at = a.view_mut().reversed_axes();
    at[[0, 1]] = 5;
    assert_eq!(at, arr2(&[[1, 5], [2, 4]]));

    let mut a = arr2(&[[1, 2, 3],
                       [4, 5, 6]]);
    let mut at = a.view_mut().reversed_axes();
    at[[2, 1]] = 7;
    assert_eq!(at, arr2(&[[1, 4], [2, 5], [3, 7]]));
}

#[test]
fn reshape() {
    let data = [1, 2, 3, 4, 5, 6, 7, 8];
    let v = aview1(&data);
    let u = v.into_shape((3, 3));
    assert!(u.is_err());
    let u = v.into_shape((2, 2, 2));
    assert!(u.is_ok());
    let u = u.unwrap();
    assert_eq!(u.shape(), &[2, 2, 2]);
    let s = u.into_shape((4, 2)).unwrap();
    assert_eq!(s.shape(), &[4, 2]);
    assert_eq!(s, aview2(&[[1, 2],
                           [3, 4],
                           [5, 6],
                           [7, 8]]));
}

#[test]
#[should_panic(expected = "IncompatibleShape")]
fn reshape_error1() {
    let data = [1, 2, 3, 4, 5, 6, 7, 8];
    let v = aview1(&data);
    let _u = v.into_shape((2, 5)).unwrap();
}

#[test]
#[should_panic(expected = "IncompatibleLayout")]
fn reshape_error2() {
    let data = [1, 2, 3, 4, 5, 6, 7, 8];
    let v = aview1(&data);
    let mut u = v.into_shape((2, 2, 2)).unwrap();
    u.swap_axes(0, 1);
    let _s = u.into_shape((2, 4)).unwrap();
}

#[test]
fn reshape_f() {
    let mut u = Array::zeros((3, 4).f());
    for (i, elt) in enumerate(u.as_slice_memory_order_mut().unwrap()) {
        *elt = i as i32;
    }
    let v = u.view();
    println!("{:?}", v);

    // noop ok
    let v2 = v.into_shape((3, 4));
    assert!(v2.is_ok());
    assert_eq!(v, v2.unwrap());

    let u = v.into_shape((3, 2, 2));
    assert!(u.is_ok());
    let u = u.unwrap();
    println!("{:?}", u);
    assert_eq!(u.shape(), &[3, 2, 2]);
    let s = u.into_shape((4, 3)).unwrap();
    println!("{:?}", s);
    assert_eq!(s.shape(), &[4, 3]);
    assert_eq!(s, aview2(&[[0, 4, 8],
                           [1, 5, 9],
                           [2, 6,10],
                           [3, 7,11]]));
}

#[test]
fn arithmetic_broadcast() {
    let mut a = arr2(&[[1., 2.], [3., 4.]]);
    let b = a.clone() * aview0(&1.);
    assert_eq!(a, b);
    a.swap_axes(0, 1);
    let b = a.clone() / aview0(&1.);
    assert_eq!(a, b);
}

#[test]
fn char_array()
{
    // test compilation & basics of non-numerical array
    let cc = RcArray::from_iter("alphabet".chars()).reshape((4, 2));
    assert!(cc.subview(Axis(1), 0) == RcArray::from_iter("apae".chars()));
}

#[test]
fn scalar_ops() {
    let a = Array::<i32, _>::zeros((5, 5));
    let b = &a + 1;
    let c = (&a + &a + 2) - 3;
    println!("{:?}", b);
    println!("{:?}", c);

    let a = Array::<f32, _>::zeros((2, 2));
    let b = (1. + a) * 3.;
    assert_eq!(b, arr2(&[[3., 3.], [3., 3.]]));

    let a = arr1(&[false, true, true]);
    let b = &a ^ true;
    let c = true ^ &a;
    assert_eq!(b, c);
    assert_eq!(true & &a, a);
    assert_eq!(b, arr1(&[true, false, false]));
    assert_eq!(true ^ &a, !a);

    let zero = Array::<f32, _>::zeros((2, 2));
    let one = &zero + 1.;
    assert_eq!(0. * &one, zero);
    assert_eq!(&one * 0., zero);
    assert_eq!((&one + &one).scalar_sum(), 8.);
    assert_eq!(&one / 2., 0.5 * &one);
    assert_eq!(&one % 1., zero);

    let zero = Array::<i32, _>::zeros((2, 2));
    let one = &zero + 1;
    assert_eq!(one.clone() << 3, 8 * &one);
    assert_eq!(3 << one.clone() , 6 * &one);

    assert_eq!(&one << 3, 8 * &one);
    assert_eq!(3 << &one , 6 * &one);
}

#[test]
fn deny_wraparound_from_vec() {
    let five = vec![0; 5];
    let five_large = Array::from_shape_vec((3, 7, 29, 36760123, 823996703), five.clone());
    assert!(five_large.is_err());
    let six = Array::from_shape_vec(6, five.clone());
    assert!(six.is_err());
}

#[should_panic]
#[test]
fn deny_wraparound_zeros() {
    //2^64 + 5 = 18446744073709551621 = 3×7×29×36760123×823996703  (5 distinct prime factors)
    let _five_large = Array::<f32, _>::zeros((3, 7, 29, 36760123, 823996703));
}

#[should_panic]
#[test]
fn deny_wraparound_reshape() {
    //2^64 + 5 = 18446744073709551621 = 3×7×29×36760123×823996703  (5 distinct prime factors)
    let five = Array::<f32, _>::zeros(5);
    let _five_large = five.into_shape((3, 7, 29, 36760123, 823996703)).unwrap();
}

#[test]
fn split_at() {
    let mut a = arr2(&[[1., 2.], [3., 4.]]);

    {
        let (c0, c1) = a.view().split_at(Axis(1), 1);

        assert_eq!(c0, arr2(&[[1.], [3.]]));
        assert_eq!(c1, arr2(&[[2.], [4.]]));
    }

    {
        let (mut r0, mut r1) = a.view_mut().split_at(Axis(0), 1);
        r0[[0, 1]] = 5.;
        r1[[0, 0]] = 8.;
    }
    assert_eq!(a, arr2(&[[1., 5.], [8., 4.]]));


    let b = RcArray::linspace(0., 59., 60).reshape((3, 4, 5));

    let (left, right) = b.view().split_at(Axis(2), 2);
    assert_eq!(left.shape(), [3, 4, 2]);
    assert_eq!(right.shape(), [3, 4, 3]);
    assert_eq!(left, arr3(&[[[0., 1.], [5., 6.], [10., 11.], [15., 16.]],
                            [[20., 21.], [25., 26.], [30., 31.], [35., 36.]],
                            [[40., 41.], [45., 46.], [50., 51.], [55., 56.]]]));

    // we allow for an empty right view when index == dim[axis]
    let (_, right) = b.view().split_at(Axis(1), 4);
    assert_eq!(right.shape(), [3, 0, 5]);
}

#[test]
#[should_panic]
fn deny_split_at_axis_out_of_bounds() {
    let a = arr2(&[[1., 2.], [3., 4.]]);
    a.view().split_at(Axis(2), 0);
}

#[test]
#[should_panic]
fn deny_split_at_index_out_of_bounds() {
    let a = arr2(&[[1., 2.], [3., 4.]]);
    a.view().split_at(Axis(1), 3);
}

#[test]
fn test_range() {
    let a = Array::range(0., 5., 1.);
    assert_eq!(a.len(), 5);
    assert_eq!(a[0],  0.);
    assert_eq!(a[4],  4.);

    let b = Array::range(0., 2.2, 1.);
    assert_eq!(b.len(), 3);
    assert_eq!(b[0],  0.);
    assert_eq!(b[2],  2.);

    let c = Array::range(0., 5., 2.);
    assert_eq!(c.len(), 3);
    assert_eq!(c[0], 0.);
    assert_eq!(c[1], 2.);
    assert_eq!(c[2], 4.);

    let d = Array::range(1.0, 2.2, 0.1);
    assert_eq!(d.len(), 13);
    assert_eq!(d[0], 1.);
    assert_eq!(d[10], 2.);
    assert_eq!(d[12], 2.2);
}

#[test]
fn test_f_order() {
    // Test that arrays are logically equal in every way,
    // even if the underlying memory order is different
    let c = arr2(&[[1, 2, 3],
                   [4, 5, 6]]);
    let mut f = Array::zeros(c.dim().f());
    f.assign(&c);
    assert_eq!(f, c);
    assert_eq!(f.shape(), c.shape());
    assert_eq!(c.strides(), &[3, 1]);
    assert_eq!(f.strides(), &[1, 2]);
    itertools::assert_equal(f.iter(), c.iter());
    itertools::assert_equal(f.inner_iter(), c.inner_iter());
    itertools::assert_equal(f.outer_iter(), c.outer_iter());
    itertools::assert_equal(f.axis_iter(Axis(0)), c.axis_iter(Axis(0)));
    itertools::assert_equal(f.axis_iter(Axis(1)), c.axis_iter(Axis(1)));

    let dupc = &c + &c;
    let dupf = &f + &f;
    assert_eq!(dupc, dupf);
}

#[test]
fn to_owned_memory_order() {
    // check that .to_owned() makes f-contiguous arrays out of f-contiguous
    // input.
    let c = arr2(&[[1, 2, 3],
                   [4, 5, 6]]);
    let mut f = c.view();
    f.swap_axes(0, 1);
    let fo = f.to_owned();
    assert_eq!(f, fo);
    assert_eq!(f.strides(), fo.strides());
}

#[test]
fn map_memory_order() {
    let a = arr3(&[[[1, 2, 3],
                    [4, 5, 6]],
                   [[7, 8, 9],
                    [0, -1, -2]]]);
    let mut v = a.view();
    v.swap_axes(0, 1);
    let amap = v.map(|x| *x >= 3);
    assert_eq!(amap.dim(), v.dim());
    assert_eq!(amap.strides(), v.strides());
}

#[test]
fn test_contiguous() {
    let c = arr3(&[[[1, 2, 3],
                    [4, 5, 6]],
                   [[4, 5, 6],
                    [7, 7, 7]]]);
    assert!(c.is_standard_layout());
    assert!(c.as_slice_memory_order().is_some());
    let v = c.slice(s![.., 0..1, ..]);
    assert!(!v.is_standard_layout());
    assert!(!v.as_slice_memory_order().is_some());

    let v = c.slice(s![1..2, .., ..]);
    assert!(v.is_standard_layout());
    assert!(v.as_slice_memory_order().is_some());
    let v = v.reversed_axes();
    assert!(!v.is_standard_layout());
    assert!(v.as_slice_memory_order().is_some());
    let mut v = v.reversed_axes();
    v.swap_axes(1, 2);
    assert!(!v.is_standard_layout());
    assert!(v.as_slice_memory_order().is_some());

    let a = Array::<f32, _>::zeros((20, 1));
    let b = Array::<f32, _>::zeros((20, 1).f());
    assert!(a.as_slice().is_some());
    assert!(b.as_slice().is_some());
    assert!(a.as_slice_memory_order().is_some());
    assert!(b.as_slice_memory_order().is_some());
    let a = a.t();
    let b = b.t();
    assert!(a.as_slice().is_some());
    assert!(b.as_slice().is_some());
    assert!(a.as_slice_memory_order().is_some());
    assert!(b.as_slice_memory_order().is_some());
}

#[test]
fn test_all_close() {
    let c = arr3(&[[[1., 2., 3.],
                    [1.5, 1.5, 3.]],
                   [[1., 2., 3.],
                    [1., 2.5, 3.]]]);
    assert!(c.all_close(&aview1(&[1., 2., 3.]), 1.));
    assert!(!c.all_close(&aview1(&[1., 2., 3.]), 0.1));
}

#[test]
fn test_swap() {
    let mut a = arr2(&[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]);
    let b = a.clone();

    for i in 0..a.rows() {
        for j in i + 1..a.cols() {
            a.swap((i, j), (j, i));
        }
    }
    assert_eq!(a, b.t());
}

#[test]
fn test_shape() {
    let data = [0, 1, 2, 3, 4, 5];
    let a = Array::from_shape_vec((1, 2, 3), data.to_vec()).unwrap();
    let b = Array::from_shape_vec((1, 2, 3).f(), data.to_vec()).unwrap();
    let c = Array::from_shape_vec((1, 2, 3).strides((1, 3, 1)), data.to_vec()).unwrap();
    println!("{:?}", a);
    println!("{:?}", b);
    println!("{:?}", c);
    assert_eq!(a.strides(), &[6, 3, 1]);
    assert_eq!(b.strides(), &[1, 1, 2]);
    assert_eq!(c.strides(), &[1, 3, 1]);
}

#[test]
fn test_view_from_shape_ptr() {
    let data = [0, 1, 2, 3, 4, 5];
    let view = unsafe { ArrayView::from_shape_ptr((2, 3), data.as_ptr()) };
    assert_eq!(view, aview2(&[[0, 1, 2], [3, 4, 5]]));

    let mut data = data;
    let mut view = unsafe { ArrayViewMut::from_shape_ptr((2, 3), data.as_mut_ptr()) };
    view[[1, 2]] = 6;
    assert_eq!(view, aview2(&[[0, 1, 2], [3, 4, 6]]));
    view[[0, 1]] = 0;
    assert_eq!(view, aview2(&[[0, 0, 2], [3, 4, 6]]));
}

#[test]
fn test_default() {
    let a = <Array<f32, Ix2> as Default>::default();
    assert_eq!(a, aview2(&[[0.0; 0]; 0]));


    #[derive(Default, Debug, PartialEq)]
    struct Foo(i32);
    let b = <Array<Foo, Ix0> as Default>::default();
    assert_eq!(b, arr0(Foo::default()));
}

#[test]
fn test_map_axis() {
    let a = arr2(&[[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10,11,12]]);

    let b = a.map_axis(Axis(0), |view| view.scalar_sum());
    let answer1 = arr1(&[22, 26, 30]);
    assert_eq!(b, answer1);
    let c = a.map_axis(Axis(1), |view| view.scalar_sum());
    let answer2 = arr1(&[6, 15, 24, 33]);
    assert_eq!(c, answer2);
}

#[test]
fn test_array_clone_unalias() {
    let a = Array2::<i32>::zeros((3, 3));
    let mut b = a.clone();
    b.fill(1);
    assert!(a != b);
    assert_eq!(a, Array2::zeros((3, 3)));
}

#[test]
fn test_array_clone_same_view() {
    let mut a = Array::from_iter(0..9).into_shape((3, 3)).unwrap();
    a.islice(s![..;-1, ..;-1]);
    let b = a.clone();
    assert_eq!(a, b);
}
