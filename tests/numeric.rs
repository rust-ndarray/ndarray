#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::float_cmp
)]

use approx::assert_abs_diff_eq;
use ndarray::{arr0, arr1, arr2, array, aview1, Array, Array1, Array2, Array3, Axis};
use std::f64;

#[test]
fn test_mean_with_nan_values() {
    let a = array![f64::NAN, 1.];
    assert!(a.mean().unwrap().is_nan());
}

#[test]
fn test_mean_with_empty_array_of_floats() {
    let a: Array1<f64> = array![];
    assert!(a.mean().is_none());
}

#[test]
fn test_mean_with_array_of_floats() {
    let a: Array1<f64> = array![
        0.99889651, 0.0150731, 0.28492482, 0.83819218, 0.48413156, 0.80710412, 0.41762936,
        0.22879429, 0.43997224, 0.23831807, 0.02416466, 0.6269962, 0.47420614, 0.56275487,
        0.78995021, 0.16060581, 0.64635041, 0.34876609, 0.78543249, 0.19938356, 0.34429457,
        0.88072369, 0.17638164, 0.60819363, 0.250392, 0.69912532, 0.78855523, 0.79140914,
        0.85084218, 0.31839879, 0.63381769, 0.22421048, 0.70760302, 0.99216018, 0.80199153,
        0.19239188, 0.61356023, 0.31505352, 0.06120481, 0.66417377, 0.63608897, 0.84959691,
        0.43599069, 0.77867775, 0.88267754, 0.83003623, 0.67016118, 0.67547638, 0.65220036,
        0.68043427
    ];
    let exact_mean = 0.5475494054;
    assert_abs_diff_eq!(a.mean().unwrap(), exact_mean);
}

#[test]
fn sum_mean() {
    let a: Array2<f64> = arr2(&[[1., 2.], [3., 4.]]);
    assert_eq!(a.sum_axis(Axis(0)), arr1(&[4., 6.]));
    assert_eq!(a.sum_axis(Axis(1)), arr1(&[3., 7.]));
    assert_eq!(a.mean_axis(Axis(0)), Some(arr1(&[2., 3.])));
    assert_eq!(a.mean_axis(Axis(1)), Some(arr1(&[1.5, 3.5])));
    assert_eq!(a.sum_axis(Axis(1)).sum_axis(Axis(0)), arr0(10.));
    assert_eq!(a.view().mean_axis(Axis(1)).unwrap(), aview1(&[1.5, 3.5]));
    assert_eq!(a.sum(), 10.);
}

#[test]
fn sum_mean_empty() {
    assert_eq!(Array3::<f32>::ones((2, 0, 3)).sum(), 0.);
    assert_eq!(Array1::<f32>::ones(0).sum_axis(Axis(0)), arr0(0.));
    assert_eq!(
        Array3::<f32>::ones((2, 0, 3)).sum_axis(Axis(1)),
        Array::zeros((2, 3)),
    );
    let a = Array1::<f32>::ones(0).mean_axis(Axis(0));
    assert_eq!(a, None);
    let a = Array3::<f32>::ones((2, 0, 3)).mean_axis(Axis(1));
    assert_eq!(a, None);
}

#[test]
#[cfg(feature = "std")]
fn var() {
    let a = array![1., -4.32, 1.14, 0.32];
    assert_abs_diff_eq!(a.var(0.), 5.049875, epsilon = 1e-8);
}

#[test]
#[cfg(feature = "std")]
#[should_panic]
fn var_negative_ddof() {
    let a = array![1., 2., 3.];
    a.var(-1.);
}

#[test]
#[cfg(feature = "std")]
#[should_panic]
fn var_too_large_ddof() {
    let a = array![1., 2., 3.];
    a.var(4.);
}

#[test]
#[cfg(feature = "std")]
fn var_nan_ddof() {
    let a = Array2::<f64>::zeros((2, 3));
    let v = a.var(::std::f64::NAN);
    assert!(v.is_nan());
}

#[test]
#[cfg(feature = "std")]
fn var_empty_arr() {
    let a: Array1<f64> = array![];
    assert!(a.var(0.0).is_nan());
}

#[test]
#[cfg(feature = "std")]
fn std() {
    let a = array![1., -4.32, 1.14, 0.32];
    assert_abs_diff_eq!(a.std(0.), 2.24719, epsilon = 1e-5);
}

#[test]
#[cfg(feature = "std")]
#[should_panic]
fn std_negative_ddof() {
    let a = array![1., 2., 3.];
    a.std(-1.);
}

#[test]
#[cfg(feature = "std")]
#[should_panic]
fn std_too_large_ddof() {
    let a = array![1., 2., 3.];
    a.std(4.);
}

#[test]
#[cfg(feature = "std")]
fn std_nan_ddof() {
    let a = Array2::<f64>::zeros((2, 3));
    let v = a.std(::std::f64::NAN);
    assert!(v.is_nan());
}

#[test]
#[cfg(feature = "std")]
fn std_empty_arr() {
    let a: Array1<f64> = array![];
    assert!(a.std(0.0).is_nan());
}

#[test]
#[cfg(feature = "approx")]
fn var_axis() {
    use ndarray::{aview0, aview2};

    let a = array![
        [
            [-9.76, -0.38, 1.59, 6.23],
            [-8.57, -9.27, 5.76, 6.01],
            [-9.54, 5.09, 3.21, 6.56],
        ],
        [
            [8.23, -9.63, 3.76, -3.48],
            [-5.46, 5.86, -2.81, 1.35],
            [-1.08, 4.66, 8.34, -0.73],
        ],
    ];
    assert_abs_diff_eq!(
        a.var_axis(Axis(0), 1.5),
        aview2(&[
            [3.236401e+02, 8.556250e+01, 4.708900e+00, 9.428410e+01],
            [9.672100e+00, 2.289169e+02, 7.344490e+01, 2.171560e+01],
            [7.157160e+01, 1.849000e-01, 2.631690e+01, 5.314410e+01]
        ]),
        epsilon = 1e-4,
    );
    assert_abs_diff_eq!(
        a.var_axis(Axis(1), 1.7),
        aview2(&[
            [0.61676923, 80.81092308, 6.79892308, 0.11789744],
            [75.19912821, 114.25235897, 48.32405128, 9.03020513],
        ]),
        epsilon = 1e-8,
    );
    assert_abs_diff_eq!(
        a.var_axis(Axis(2), 2.3),
        aview2(&[
            [79.64552941, 129.09663235, 95.98929412],
            [109.64952941, 43.28758824, 36.27439706],
        ]),
        epsilon = 1e-8,
    );

    let b = array![[1.1, 2.3, 4.7]];
    assert_abs_diff_eq!(
        b.var_axis(Axis(0), 0.),
        aview1(&[0., 0., 0.]),
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(b.var_axis(Axis(1), 0.), aview1(&[2.24]), epsilon = 1e-12);

    let c = array![[], []];
    assert_eq!(c.var_axis(Axis(0), 0.), aview1(&[]));

    let d = array![1.1, 2.7, 3.5, 4.9];
    assert_abs_diff_eq!(d.var_axis(Axis(0), 0.), aview0(&1.8875), epsilon = 1e-12);
}

#[test]
#[cfg(feature = "approx")]
fn std_axis() {
    use ndarray::aview2;

    let a = array![
        [
            [0.22935481, 0.08030619, 0.60827517, 0.73684379],
            [0.90339851, 0.82859436, 0.64020362, 0.2774583],
            [0.44485313, 0.63316367, 0.11005111, 0.08656246]
        ],
        [
            [0.28924665, 0.44082454, 0.59837736, 0.41014531],
            [0.08382316, 0.43259439, 0.1428889, 0.44830176],
            [0.51529756, 0.70111616, 0.20799415, 0.91851457]
        ],
    ];
    assert_abs_diff_eq!(
        a.std_axis(Axis(0), 1.5),
        aview2(&[
            [0.05989184, 0.36051836, 0.00989781, 0.32669847],
            [0.81957535, 0.39599997, 0.49731472, 0.17084346],
            [0.07044443, 0.06795249, 0.09794304, 0.83195211],
        ]),
        epsilon = 1e-4,
    );
    assert_abs_diff_eq!(
        a.std_axis(Axis(1), 1.7),
        aview2(&[
            [0.42698655, 0.48139215, 0.36874991, 0.41458724],
            [0.26769097, 0.18941435, 0.30555015, 0.35118674],
        ]),
        epsilon = 1e-8,
    );
    assert_abs_diff_eq!(
        a.std_axis(Axis(2), 2.3),
        aview2(&[
            [0.41117907, 0.37130425, 0.35332388],
            [0.16905862, 0.25304841, 0.39978276],
        ]),
        epsilon = 1e-8,
    );

    let b = array![[100000., 1., 0.01]];
    assert_abs_diff_eq!(
        b.std_axis(Axis(0), 0.),
        aview1(&[0., 0., 0.]),
        epsilon = 1e-12,
    );
    assert_abs_diff_eq!(
        b.std_axis(Axis(1), 0.),
        aview1(&[47140.214021552769]),
        epsilon = 1e-6,
    );

    let c = array![[], []];
    assert_eq!(c.std_axis(Axis(0), 0.), aview1(&[]));
}

#[test]
#[should_panic]
#[cfg(feature = "std")]
fn var_axis_negative_ddof() {
    let a = array![1., 2., 3.];
    a.var_axis(Axis(0), -1.);
}

#[test]
#[should_panic]
#[cfg(feature = "std")]
fn var_axis_too_large_ddof() {
    let a = array![1., 2., 3.];
    a.var_axis(Axis(0), 4.);
}

#[test]
#[cfg(feature = "std")]
fn var_axis_nan_ddof() {
    let a = Array2::<f64>::zeros((2, 3));
    let v = a.var_axis(Axis(1), ::std::f64::NAN);
    assert_eq!(v.shape(), &[2]);
    v.mapv(|x| assert!(x.is_nan()));
}

#[test]
#[cfg(feature = "std")]
fn var_axis_empty_axis() {
    let a = Array2::<f64>::zeros((2, 0));
    let v = a.var_axis(Axis(1), 0.);
    assert_eq!(v.shape(), &[2]);
    v.mapv(|x| assert!(x.is_nan()));
}

#[test]
#[should_panic]
#[cfg(feature = "std")]
fn std_axis_bad_dof() {
    let a = array![1., 2., 3.];
    a.std_axis(Axis(0), 4.);
}

#[test]
#[cfg(feature = "std")]
fn std_axis_empty_axis() {
    let a = Array2::<f64>::zeros((2, 0));
    let v = a.std_axis(Axis(1), 0.);
    assert_eq!(v.shape(), &[2]);
    v.mapv(|x| assert!(x.is_nan()));
}
