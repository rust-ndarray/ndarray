#![allow(non_snake_case)]
#![cfg_attr(feature = "assign_ops", feature(augmented_assignments))]

#[macro_use]
extern crate ndarray;

use ndarray::{Array, S, Si,
    OwnedArray,
};
use ndarray::{arr0, arr1, arr2,
    aview0,
    aview1,
    aview2,
    aview_mut1,
};
use ndarray::Indexes;

#[test]
fn test_matmul_rcarray()
{
    let mut A = Array::<usize, _>::zeros((2, 3));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let mut B = Array::<usize, _>::zeros((3, 4));
    for (i, elt) in B.iter_mut().enumerate() {
        *elt = i;
    }

    let c = A.mat_mul(&B);
    println!("A = \n{:?}", A);
    println!("B = \n{:?}", B);
    println!("A x B = \n{:?}", c);
    unsafe {
        let result = Array::from_vec_dim_unchecked((2, 4), vec![20, 23, 26, 29, 56, 68, 80, 92]);
        assert_eq!(c.shape(), result.shape());
        assert!(c.iter().zip(result.iter()).all(|(a,b)| a == b));
        assert!(c == result);
    }
}

#[test]
fn test_mat_mul() {
    // smoke test, a big matrix multiplication of uneven size
    let (n, m) = (45, 33);
    let a = Array::linspace(0., ((n * m) - 1) as f32, n as usize * m as usize ).reshape((n, m));
    let b = Array::eye(m);
    assert_eq!(a.mat_mul(&b), a);
    let c = Array::eye(n);
    assert_eq!(c.mat_mul(&a), a);
}


#[test]
fn test_slice()
{
    let mut A = Array::<usize, _>::zeros((3, 4));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let vi = A.slice(s![1.., ..;2]);
    assert_eq!(vi.dim(), (2, 2));
    let vi = A.slice(&[S, S]);
    assert_eq!(vi.shape(), A.shape());
    assert!(vi.iter().zip(A.iter()).all(|(a, b)| a == b));
}

#[should_panic]
#[test]
fn slice_oob()
{
    let a = Array::<i32, _>::zeros((3, 4));
    let _vi = a.slice(&[Si(0, Some(10), 1), S]);
}

#[test]
fn test_index()
{
    let mut A = Array::<usize, _>::zeros((2, 3));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    for ((i, j), a) in Indexes::new((2, 3)).zip(A.iter()) {
        assert_eq!(*a, A[[i, j]]);
    }

    let vi = A.slice(&[Si(1, None, 1), Si(0, None, 2)]);
    let mut it = vi.iter();
    for ((i, j), x) in Indexes::new((1, 2)).zip(it.by_ref()) {
        assert_eq!(*x, vi[[i, j]]);
    }
    assert!(it.next().is_none());
}

#[test]
fn test_add()
{
    let mut A = Array::<usize, _>::zeros((2, 2));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let B = A.clone();
    A.iadd(&B);
    assert_eq!(A[[0, 0]], 0);
    assert_eq!(A[[0, 1]], 2);
    assert_eq!(A[[1, 0]], 4);
    assert_eq!(A[[1, 1]], 6);
}

#[test]
fn test_multidim()
{
    let mut mat = Array::zeros(2*3*4*5*6).reshape((2,3,4,5,6));
    mat[(0,0,0,0,0)] = 22u8;
    {
        for (i, elt) in mat.iter_mut().enumerate() {
            *elt = i as u8;
        }
    }
    //println!("shape={:?}, strides={:?}", mat.shape(), mat.strides);
    assert_eq!(mat.dim(), (2,3,4,5,6));
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
    let mut mat = Array::zeros((2, 4, 2));
    mat[(0, 0, 0)] = 1.0f32;
    for (i, elt) in mat.iter_mut().enumerate() {
        *elt = i as f32;
    }

    {
        let vi = mat.slice(&[S, Si(0, None, -1), Si(0, None, -1)]);
        assert_eq!(vi.dim(), (2,4,2));
        // Test against sequential iterator
        let seq = [7f32,6., 5.,4.,3.,2.,1.,0.,15.,14.,13., 12.,11.,  10.,   9.,   8.];
        for (a, b) in vi.clone().iter().zip(seq.iter()) {
            assert_eq!(*a, *b);
        }
    }
    {
        let vi = mat.slice(&[S, Si(0, None, -5), S]);
        let seq = [6_f32, 7., 14., 15.];
        for (a, b) in vi.iter().zip(seq.iter()) {
            assert_eq!(*a, *b);
        }
    }
}

#[test]
fn test_cow()
{
    let mut mat = Array::<isize, _>::zeros((2,2));
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
    let mat = Array::linspace(0., 15., 16).reshape((2, 4, 2));
    let s1 = mat.subview(0,0);
    let s2 = mat.subview(0,1);
    assert_eq!(s1.dim(), (4, 2));
    assert_eq!(s2.dim(), (4, 2));
    let n = Array::linspace(8., 15., 8).reshape((4,2));
    assert_eq!(n, s2);
    let m = Array::from_vec(vec![2., 3., 10., 11.]).reshape((2, 2));
    assert_eq!(m, mat.subview(1, 1));
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
    let d = Array::<f32, _>::zeros(()).into_diag();
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
    assert!(a.raw_data() == [1., 2., 3., 4.]);
    assert!(b.raw_data() == [1., 3., 2., 4.]);
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
    let x1 = a.subview(0, 0);
    assert!(x1.is_standard_layout());
    let x2 = a.subview(1, 0);
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
    a.assign(&Array::zeros(1));
    assert_eq!(a, Array::zeros((2, 2)));

    /* Test other type */
    a.assign(&OwnedArray::from_elem((2, 2), 3.));
    assert_eq!(a, Array::from_elem((2, 2), 3.));

    /* Test mut view */
    let mut a = arr2(&[[1, 2], [3, 4]]);
    {
        let mut v = a.view_mut();
        v.islice(&[Si(0, Some(1), 1), S]);
        v.assign_scalar(&0);
    }
    assert_eq!(a, arr2(&[[0, 0], [3, 4]]));
}

#[test]
fn sum_mean()
{
    let a = arr2(&[[1., 2.], [3., 4.]]);
    assert_eq!(a.sum(0), arr1(&[4., 6.]));
    assert_eq!(a.sum(1), arr1(&[3., 7.]));
    assert_eq!(a.mean(0), arr1(&[2., 3.]));
    assert_eq!(a.mean(1), arr1(&[1.5, 3.5]));
    assert_eq!(a.sum(1).sum(0), arr0(10.));
    assert_eq!(a.view().mean(1), aview1(&[1.5, 3.5]));
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
    let a = arr1::<f32>(&[]);
    for _ in a.iter() {
        assert!(false);
    }
    println!("{:?}", a);
    let b = arr2::<f32, _>(&[[], [], [], []]);
    println!("{:?}\n{:?}", b.shape(), b);

    // we can even get a subarray of b
    let bsub = b.subview(0, 2);
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
    let c = a.map(|x| x).into_shared();
    assert_eq!(a[(0, 0)], *c[(0, 0)]);
}

#[test]
fn raw_data_mut()
{
    let a = arr2(&[[1., 2.], [3., 4.0f32]]);
    let mut b = a.clone();
    for elt in b.raw_data_mut() {
        *elt = 0.;
    }
    assert!(a != b, "{:?} != {:?}", a, b);
}

#[test]
fn owned_array1() {
    let mut a = OwnedArray::from_vec(vec![1, 2, 3, 4]);
    for elt in a.iter_mut() {
        *elt = 2;
    }
    for elt in a.iter() {
        assert_eq!(*elt, 2);
    }
    assert_eq!(a.shape(), &[4]);

    let mut a = OwnedArray::zeros((2, 2));
    let mut b = Array::zeros((2, 2));
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

    let a = OwnedArray::from_vec_dim_stride(dim, strides, v).unwrap();
    assert_eq!(a.strides(), &[1, 4, 2]);
}

#[test]
fn views() {
    let a = Array::from_vec(vec![1, 2, 3, 4]).reshape((2, 2));
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
    let mut a = Array::from_vec(vec![1, 2, 3, 4]).reshape((2, 2));
    for elt in &mut a.view_mut() {
        *elt = 0;
    }
    assert_eq!(a, OwnedArray::zeros((2, 2)));
    {
        let mut b = a.view_mut();
        b[(0, 0)] = 7;
    }
    assert_eq!(a[(0, 0)], 7);

    for elt in a.view_mut() {
        *elt = 2;
    }
    assert_eq!(a, Array::from_elem((2, 2), 2));
}

#[test]
fn slice_mut() {
    let mut a = Array::from_vec(vec![1, 2, 3, 4]).reshape((2, 2));
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

#[cfg(feature = "assign_ops")]
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
        a.slice_mut(&[Si(0, Some(2), 1), Si(0, None, 2)])
         .iadd_scalar(&1);
        println!("{}", a);
    }
    assert_eq!(data, [1, 0, 1, 0,  1, 0, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0]);
}

#[test]
fn transpose_view() {
    let a = arr2(&[[1, 2],
                   [3, 4]]);
    let at = a.transpose_view();
    assert_eq!(at, arr2(&[[1, 3], [2, 4]]));

    let a = arr2(&[[1, 2, 3],
                   [4, 5, 6]]);
    let at = a.transpose_view();
    assert_eq!(at, arr2(&[[1, 4], [2, 5], [3, 6]]));
}

#[test]
fn transpose_view_mut() {
    let mut a = arr2(&[[1, 2],
                       [3, 4]]);
    let mut at = a.transpose_view_mut();
    at[[0, 1]] = 5;
    assert_eq!(at, arr2(&[[1, 5], [2, 4]]));

    let mut a = arr2(&[[1, 2, 3],
                       [4, 5, 6]]);
    let mut at = a.transpose_view_mut();
    at[[2, 1]] = 7;
    assert_eq!(at, arr2(&[[1, 4], [2, 5], [3, 7]]));
}

#[test]
fn transpose_into() {
    let a = arr2(&[[1, 2],
                   [3, 4]]);
    let a_ = a.clone();

    let at = a.transpose_into();

    assert_eq!(at, a_.transpose_view());
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
#[should_panic(expected = "IncompatibleShapes")]
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
    let mut u = v.into_shape((4, 2)).unwrap();
    u.swap_axes(0, 1);
    let _s = u.into_shape((2, 4)).unwrap();
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
    let cc = Array::from_iter("alphabet".chars()).reshape((4, 2));
    assert!(cc.subview(1, 0) == Array::from_iter("apae".chars()));
}

#[test]
fn scalar_ops() {
    let a = OwnedArray::<i32, _>::zeros((5, 5));
    let b = &a + 1;
    let c = (&a + &a + 2) - 3;
    println!("{:?}", b);
    println!("{:?}", c);

    let a = OwnedArray::<f32, _>::zeros((2, 2));
    let b = (1. + a) * 3.;
    assert_eq!(b, arr2(&[[3., 3.], [3., 3.]]));

    let a = arr1(&[false, true, true]);
    let b = &a ^ true;
    let c = true ^ &a;
    assert_eq!(b, c);
    assert_eq!(true & &a, a);
    assert_eq!(b, arr1(&[true, false, false]));
    assert_eq!(true ^ &a, !a);

    let zero = OwnedArray::<f32, _>::zeros((2, 2));
    let one = &zero + 1.;
    assert_eq!(0. * &one, zero);
    assert_eq!(&one * 0., zero);
    assert_eq!((&one + &one).scalar_sum(), 8.);
    assert_eq!(&one / 2., 0.5 * &one);
    assert_eq!(&one % 1., zero);

    let zero = OwnedArray::<i32, _>::zeros((2, 2));
    let one = &zero + 1;
    assert_eq!(one.clone() << 3, 8 * &one);
    assert_eq!(3 << one.clone() , 6 * &one);

    assert_eq!(&one << 3, 8 * &one);
    assert_eq!(3 << &one , 6 * &one);
}

#[test]
fn deny_wraparound_from_vec() {
    let five = vec![0; 5];
    let _five_large = OwnedArray::from_vec_dim((3, 7, 29, 36760123, 823996703), five.clone());
    assert!(_five_large.is_err());
    let six = OwnedArray::from_vec_dim(6, five.clone());
    assert!(six.is_err());
}

#[should_panic]
#[test]
fn deny_wraparound_zeros() {
    //2^64 + 5 = 18446744073709551621 = 3×7×29×36760123×823996703  (5 distinct prime factors)
    let _five_large = OwnedArray::<f32, _>::zeros((3, 7, 29, 36760123, 823996703));
}

#[should_panic]
#[test]
fn deny_wraparound_reshape() {
    //2^64 + 5 = 18446744073709551621 = 3×7×29×36760123×823996703  (5 distinct prime factors)
    let five = OwnedArray::<f32, _>::zeros(5);
    let _five_large = five.into_shape((3, 7, 29, 36760123, 823996703)).unwrap();
}
