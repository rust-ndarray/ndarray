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
