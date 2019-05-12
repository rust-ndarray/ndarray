extern crate defmac;
extern crate ndarray;

use defmac::defmac;
use ndarray::prelude::*;

#[test]
fn test_from_shape_fn() {
    let step = 3.1;
    let h = Array::from_shape_fn((5, 5), |(i, j)| {
        f64::sin(i as f64 / step) * f64::cos(j as f64 / step)
    });
    assert_eq!(h.shape(), &[5, 5]);
}

#[test]
fn test_dimension_zero() {
    let a: Array2<f32> = Array2::from(vec![[], [], []]);
    assert_eq!(vec![0.; 0], a.into_raw_vec());
    let a: Array3<f32> = Array3::from(vec![[[]], [[]], [[]]]);
    assert_eq!(vec![0.; 0], a.into_raw_vec());
}

#[test]
#[cfg(feature = "approx")]
fn test_arc_into_owned() {
    use approx::assert_abs_diff_ne;

    let a = Array2::from_elem((5, 5), 1.).into_shared();
    let mut b = a.clone();
    b.fill(0.);
    let mut c = b.into_owned();
    c.fill(2.);
    // test that they are unshared
    assert_abs_diff_ne!(a, c, epsilon = 0.01);
}

#[test]
fn test_arcarray_thread_safe() {
    fn is_send<T: Send>(_t: &T) {}
    fn is_sync<T: Sync>(_t: &T) {}
    let a = Array2::from_elem((5, 5), 1.).into_shared();

    is_send(&a);
    is_sync(&a);
}

#[test]
fn test_uninit() {
    unsafe {
        let mut a = Array::<f32, _>::uninitialized((3, 4).f());
        assert_eq!(a.dim(), (3, 4));
        assert_eq!(a.strides(), &[1, 3]);
        let b = Array::<f32, _>::linspace(0., 25., a.len())
            .into_shape(a.dim())
            .unwrap();
        a.assign(&b);
        assert_eq!(&a, &b);
        assert_eq!(a.t(), b.t());
    }
}

#[test]
fn test_from_fn_c0() {
    let a = Array::from_shape_fn((), |i| i);
    assert_eq!(a[()], ());
    assert_eq!(a.len(), 1);
    assert_eq!(a.shape(), &[]);
}

#[test]
fn test_from_fn_c1() {
    let a = Array::from_shape_fn(28, |i| i);
    for (i, elt) in a.indexed_iter() {
        assert_eq!(i, *elt);
    }
}

#[test]
fn test_from_fn_c() {
    let a = Array::from_shape_fn((4, 7), |i| i);
    for (i, elt) in a.indexed_iter() {
        assert_eq!(i, *elt);
    }
}

#[test]
fn test_from_fn_c3() {
    let a = Array::from_shape_fn((4, 3, 7), |i| i);
    for (i, elt) in a.indexed_iter() {
        assert_eq!(i, *elt);
    }
}

#[test]
fn test_from_fn_f0() {
    let a = Array::from_shape_fn(().f(), |i| i);
    assert_eq!(a[()], ());
    assert_eq!(a.len(), 1);
    assert_eq!(a.shape(), &[]);
}

#[test]
fn test_from_fn_f1() {
    let a = Array::from_shape_fn(28.f(), |i| i);
    for (i, elt) in a.indexed_iter() {
        assert_eq!(i, *elt);
    }
}

#[test]
fn test_from_fn_f() {
    let a = Array::from_shape_fn((4, 7).f(), |i| i);
    for (i, elt) in a.indexed_iter() {
        assert_eq!(i, *elt);
    }
}

#[test]
fn test_from_fn_f_with_zero() {
    defmac!(test_from_fn_f_with_zero shape => {
        let a = Array::from_shape_fn(shape.f(), |i| i);
        assert_eq!(a.len(), 0);
        assert_eq!(a.shape(), &shape);
    });
    test_from_fn_f_with_zero!([0]);
    test_from_fn_f_with_zero!([0, 1]);
    test_from_fn_f_with_zero!([2, 0]);
    test_from_fn_f_with_zero!([0, 1, 2]);
    test_from_fn_f_with_zero!([2, 0, 1]);
    test_from_fn_f_with_zero!([1, 2, 0]);
}

#[test]
fn test_from_fn_f3() {
    let a = Array::from_shape_fn((4, 2, 7).f(), |i| i);
    for (i, elt) in a.indexed_iter() {
        assert_eq!(i, *elt);
    }
}

#[test]
fn deny_wraparound_from_vec() {
    let five = vec![0; 5];
    let five_large = Array::from_shape_vec((3, 7, 29, 36760123, 823996703), five.clone());
    assert!(five_large.is_err());
    let six = Array::from_shape_vec(6, five.clone());
    assert!(six.is_err());
}

#[test]
fn test_ones() {
    let mut a = Array::<f32, _>::zeros((2, 3, 4));
    a.fill(1.0);
    let b = Array::<f32, _>::ones((2, 3, 4));
    assert_eq!(a, b);
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

#[should_panic]
#[test]
fn deny_wraparound_default() {
    let _five_large = Array::<f32, _>::default((3, 7, 29, 36760123, 823996703));
}

#[should_panic]
#[test]
fn deny_wraparound_from_shape_fn() {
    let _five_large = Array::<f32, _>::from_shape_fn((3, 7, 29, 36760123, 823996703), |_| 0.);
}

#[should_panic]
#[test]
fn deny_wraparound_uninit() {
    unsafe {
        let _five_large = Array::<f32, _>::uninitialized((3, 7, 29, 36760123, 823996703));
    }
}
