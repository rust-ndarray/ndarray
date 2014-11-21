
extern crate ndarray;

use ndarray::{arr0, arr1};

#[test]
fn formatting()
{
    let a = arr1::<f32>(&[1., 2., 3., 4.]);
    assert_eq!(a.to_string().as_slice(),
               //"[   1,    2,    3,    4]");
               "[1, 2, 3, 4]");
    let a = a.reshape((4, 1, 1));
    assert_eq!(a.to_string().as_slice(),
               //"[[[   1]],\n [[   2]],\n [[   3]],\n [[   4]]]");
               "[[[1]],\n [[2]],\n [[3]],\n [[4]]]");
    let a = a.reshape((2, 2));
    assert_eq!(a.to_string().as_slice(),
               //"[[   1,    2],\n [   3,    4]]");
               "[[1, 2],\n [3, 4]]");
    assert_eq!(format!("{:#4}", a).as_slice(),
               "[[   1,    2], [   3,    4]]");

    let b = arr0::<f32>(3.5);
    assert_eq!(b.to_string().as_slice(),
               "3.5");

    let c = arr1::<f32>(&[1.1, 2.2, 33., 440.]);
    let s = format!("{:.3e}", c);
    assert_eq!(s.as_slice(),
               "[1.100e0, 2.200e0, 3.300e1, 4.400e2]");
}
