
extern crate ndarray;

use ndarray::{arr0, arr1};

#[test]
fn formatting()
{
    let a = arr1::<f32>([1., 2., 3., 4.]);
    assert_eq!(a.to_string().as_slice(),
               "[   1,    2,    3,    4]");
    let a = a.reshape((4u, 1u, 1u));
    assert_eq!(a.to_string().as_slice(),
               "[[[   1]],\n [[   2]],\n [[   3]],\n [[   4]]]");
    let a = a.reshape((2u, 2u));
    assert_eq!(a.to_string().as_slice(),
               "[[   1,    2],\n [   3,    4]]");

    let b = arr0::<f32>(3.5);
    assert_eq!(b.to_string().as_slice(),
               "3.5");
}
