extern crate approx;
use std::f64;
use ndarray::{Array1, array};
use approx::abs_diff_eq;

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
            0.99889651, 0.0150731 , 0.28492482, 0.83819218, 0.48413156,
            0.80710412, 0.41762936, 0.22879429, 0.43997224, 0.23831807,
            0.02416466, 0.6269962 , 0.47420614, 0.56275487, 0.78995021,
            0.16060581, 0.64635041, 0.34876609, 0.78543249, 0.19938356,
            0.34429457, 0.88072369, 0.17638164, 0.60819363, 0.250392  ,
            0.69912532, 0.78855523, 0.79140914, 0.85084218, 0.31839879,
            0.63381769, 0.22421048, 0.70760302, 0.99216018, 0.80199153,
            0.19239188, 0.61356023, 0.31505352, 0.06120481, 0.66417377,
            0.63608897, 0.84959691, 0.43599069, 0.77867775, 0.88267754,
            0.83003623, 0.67016118, 0.67547638, 0.65220036, 0.68043427
        ];
    // Computed using NumPy
    let expected_mean = 0.5475494059146699;
    abs_diff_eq!(a.mean().unwrap(), expected_mean, epsilon = f64::EPSILON);
}