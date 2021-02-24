#![cfg(feature = "rayon")]
use ndarray::{Array, arr2, Array4};
use ndarray::linalg::ParDot;

#[test]
fn test_muti_par_dot() {
    let a = Array::range(0., 3., 1.);
    let b = Array::range(0., 12., 1.).into_shape((2, 3, 2)).unwrap();
    let c =a.par_dot(&b);
    assert_eq!(c, arr2(&[[10.0, 13.0], [28.0, 31.0]]));

    let a = Array::range(0., 6., 1.).into_shape((2, 3)).unwrap();
    let b = Array::range(0., 24., 1.).into_shape((2, 2, 3, 2)).unwrap();
    let c =a.par_dot(&b);
    let v = vec![10.0, 13.0, 28.0, 31.0, 46.0, 49.0, 64.0, 67.0, 28.0, 40.0, 100.0, 112.0, 172.0, 184.0, 244.0, 256.0];
    assert_eq!(c, Array4::from_shape_vec((2, 2, 2, 2), v).unwrap());

    let a = Array::range(0., 6., 1.).into_shape((2, 3)).unwrap();
    let v =vec![1, 2, 1, 2, 1, 3, 2];
    let b = Array::range(0., 24., 1.).into_shape(v.clone()).unwrap();
    let c =a.par_dot(&b);
    let v2 = vec![10.0, 13.0, 28.0, 31.0, 46.0, 49.0, 64.0, 67.0, 28.0, 40.0, 100.0, 112.0, 172.0, 184.0, 244.0, 256.0];
    assert_eq!(c, Array::from_shape_vec(vec![2, 1, 2, 1, 2, 1, 2], v2).unwrap());

    let a = Array::range(0., 6., 1.).into_shape(vec![2, 3]).unwrap();
    let v =vec![1, 2, 1, 2, 1, 3, 2];
    let b = Array::range(0., 24., 1.).into_shape(v.clone()).unwrap();
    let c =a.par_dot(&b);
    let v2 = vec![10.0, 13.0, 28.0, 31.0, 46.0, 49.0, 64.0, 67.0, 28.0, 40.0, 100.0, 112.0, 172.0, 184.0, 244.0, 256.0];
    assert_eq!(c, Array::from_shape_vec(vec![2, 1, 2, 1, 2, 1, 2], v2).unwrap());
}