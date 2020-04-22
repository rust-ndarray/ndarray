use ndarray::prelude::*;
use ndarray::rcarr1;

#[test]
fn formatting() {
    let a = rcarr1::<f32>(&[1., 2., 3., 4.]);
    assert_eq!(format!("{}", a), "[1, 2, 3, 4]");
    assert_eq!(format!("{:4}", a), "[   1,    2,    3,    4]");
    let a = a.reshape((4, 1, 1));
    assert_eq!(
        format!("{}", a),
        "\
[[[1]],

 [[2]],

 [[3]],

 [[4]]]"
    );
    assert_eq!(
        format!("{:4}", a),
        "\
[[[   1]],

 [[   2]],

 [[   3]],

 [[   4]]]",
    );

    let a = a.reshape((2, 2));
    assert_eq!(
        format!("{}", a),
        "\
[[1, 2],
 [3, 4]]"
    );
    assert_eq!(
        format!("{:4}", a),
        "\
[[   1,    2],
 [   3,    4]]"
    );

    let b = arr0::<f32>(3.5);
    assert_eq!(format!("{}", b), "3.5");

    let s = format!("{:.3e}", aview1::<f32>(&[1.1, 2.2, 33., 440.]));
    assert_eq!(s, "[1.100e0, 2.200e0, 3.300e1, 4.400e2]");

    let s = format!("{:02x}", aview1::<u8>(&[1, 0xff, 0xfe]));
    assert_eq!(s, "[01, ff, fe]");
}

#[test]
fn debug_format() {
    let a = Array2::<i32>::zeros((3, 4));
    assert_eq!(
        format!("{:?}", a),
        "\
[[0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0]], shape=[3, 4], strides=[4, 1], layout=Cc (0x5), const ndim=2"
    );
    assert_eq!(
        format!("{:?}", a.into_dyn()),
        "\
[[0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0]], shape=[3, 4], strides=[4, 1], layout=Cc (0x5), dynamic ndim=2"
    );
}
