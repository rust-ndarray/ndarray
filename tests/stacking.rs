use ndarray::{arr2, arr3, aview1, aview2, concatenate, stack, Array2, Axis, ErrorKind, Ix1};

#[test]
fn concatenating() {
    let a = arr2(&[[2., 2.], [3., 3.]]);
    let b = ndarray::concatenate(Axis(0), &[a.view(), a.view()]).unwrap();
    assert_eq!(b, arr2(&[[2., 2.], [3., 3.], [2., 2.], [3., 3.]]));

    let c = concatenate![Axis(0), a, b];
    assert_eq!(
        c,
        arr2(&[[2., 2.], [3., 3.], [2., 2.], [3., 3.], [2., 2.], [3., 3.]])
    );

    let d = concatenate![Axis(0), a.row(0), &[9., 9.]];
    assert_eq!(d, aview1(&[2., 2., 9., 9.]));

    let d = concatenate![Axis(1), a.row(0).insert_axis(Axis(1)), aview1(&[9., 9.]).insert_axis(Axis(1))];
    assert_eq!(d, aview2(&[[2., 9.],
                           [2., 9.]]));

    let d = concatenate![Axis(0), a.row(0).insert_axis(Axis(1)), aview1(&[9., 9.]).insert_axis(Axis(1))];
    assert_eq!(d, aview2(&[[2.], [2.], [9.], [9.]]));

    let res = ndarray::concatenate(Axis(1), &[a.view(), c.view()]);
    assert_eq!(res.unwrap_err().kind(), ErrorKind::IncompatibleShape);

    let res = ndarray::concatenate(Axis(2), &[a.view(), c.view()]);
    assert_eq!(res.unwrap_err().kind(), ErrorKind::OutOfBounds);

    let res: Result<Array2<f64>, _> = ndarray::concatenate(Axis(0), &[]);
    assert_eq!(res.unwrap_err().kind(), ErrorKind::Unsupported);
}

#[test]
fn stacking() {
    let a = arr2(&[[2., 2.], [3., 3.]]);
    let b = ndarray::stack(Axis(0), &[a.view(), a.view()]).unwrap();
    assert_eq!(b, arr3(&[[[2., 2.], [3., 3.]], [[2., 2.], [3., 3.]]]));

    let c = stack![Axis(0), a, a];
    assert_eq!(c, arr3(&[[[2., 2.], [3., 3.]], [[2., 2.], [3., 3.]]]));

    let c = arr2(&[[3., 2., 3.], [2., 3., 2.]]);
    let res = ndarray::stack(Axis(1), &[a.view(), c.view()]);
    assert_eq!(res.unwrap_err().kind(), ErrorKind::IncompatibleShape);

    let res = ndarray::stack(Axis(3), &[a.view(), a.view()]);
    assert_eq!(res.unwrap_err().kind(), ErrorKind::OutOfBounds);

    let res: Result<Array2<f64>, _> = ndarray::stack::<_, Ix1>(Axis(0), &[]);
    assert_eq!(res.unwrap_err().kind(), ErrorKind::Unsupported);
}
