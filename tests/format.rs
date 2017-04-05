
extern crate ndarray;

use ndarray::{arr0, rcarr1, aview1};

#[test]
fn formatting()
{
    let a = rcarr1::<f32>(&[1., 2., 3., 4.]);
    assert_eq!(format!("{}", a),
               //"[   1,    2,    3,    4]");
               "[1, 2, 3, 4]");
    assert_eq!(format!("{:4}", a),
               "[   1,    2,    3,    4]");
    let a = a.reshape((4, 1, 1));
    assert_eq!(format!("{:4}", a),
               "[[[   1]],\n [[   2]],\n [[   3]],\n [[   4]]]");

    let a = a.reshape((2, 2));
    assert_eq!(format!("{}", a), 
               "[[1, 2],\n [3, 4]]");
    assert_eq!(format!("{}", a),
               "[[1, 2],\n [3, 4]]");
    assert_eq!(format!("{:4}", a),
               "[[   1,    2],\n [   3,    4]]");

    let b = arr0::<f32>(3.5);
    assert_eq!(format!("{}", b),
               "3.5");

    let s = format!("{:.3e}", aview1::<f32>(&[1.1, 2.2, 33., 440.]));
    assert_eq!(s,
               "[1.100e0, 2.200e0, 3.300e1, 4.400e2]");

    let s = format!("{:02x}", aview1::<u8>(&[1, 0xff, 0xfe]));
    assert_eq!(s, "[01, ff, fe]");
}
