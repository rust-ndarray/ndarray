extern crate test;
extern crate ndarray;

use ndarray::Array;

#[test]
fn diag()
{
    let d = Array::from_slices([[1., 2., 3.0f32]]).diag();
    assert_eq!(d.shape(), &[1]);
    let d = Array::from_slices([[1., 2., 3.0f32], [0., 0., 0.]]).diag();
    assert_eq!(d.shape(), &[2]);
    let d = Array::<f32>::from_slices([]).diag();
    assert_eq!(d.shape(), &[0]);
    let d = Array::<f32, _>::zeros(()).diag();
    assert_eq!(d.shape(), &[1]);
}

#[test]
fn swapaxes()
{
    let mut a = Array::from_slices([[1., 2.], [3., 4.0f32]]);
    let     b = Array::from_slices([[1., 3.], [2., 4.0f32]]);
    assert!(a != b);
    a.swap_axes(0, 1);
    assert_eq!(a, b);
    a.swap_axes(1, 1);
    assert_eq!(a, b);
    assert_eq!(a.raw_data(), &[1., 2., 3., 4.]);
    assert_eq!(b.raw_data(), &[1., 3., 2., 4.]);
}

#[test]
fn standard_layout()
{
    let mut a = Array::from_slices([[1., 2.], [3., 4.0f32]]);
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
    let mut a = Array::from_slices([[1., 2.], [3., 4.0f32]]);
    let     b = Array::from_slices([[1., 3.], [2., 4.0f32]]);
    a.assign(&b);
    assert_eq!(a, b);

    /* Test broadcasting */
    a.assign(&Array::zeros(1u));
    assert_eq!(a, Array::zeros((2u, 2u)));
}

#[test]
fn dyn_dimension()
{
    let a = Array::from_slices([[1., 2.], [3., 4.0f32]]).reshape(vec![2u, 2]);
    assert_eq!(a - a, Array::zeros(vec![2u,2u]));

    let mut dim = Vec::from_elem(1024, 1u);
    dim.as_mut_slice()[16] = 4;
    dim.as_mut_slice()[17] = 3;
    let z = Array::<f32, Vec<uint>>::zeros(dim.clone());
    assert_eq!(z.shape(), dim.as_slice());
}
