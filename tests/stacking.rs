use ndarray::{arr2, arr3, aview1, stack, Array2, Axis, ErrorKind, Ix1};

#[test]
fn concatenating() {
    let a = arr2(&[[2., 2.], [3., 3.]]);
    let b = ndarray::stack(Axis(0), &[a.view(), a.view()]).unwrap();
    assert_eq!(b, arr2(&[[2., 2.], [3., 3.], [2., 2.], [3., 3.]]));

    let c = stack![Axis(0), a, b];
    assert_eq!(
        c,
        arr2(&[[2., 2.], [3., 3.], [2., 2.], [3., 3.], [2., 2.], [3., 3.]])
    );

    let d = stack![Axis(0), a.row(0), &[9., 9.]];
    assert_eq!(d, aview1(&[2., 2., 9., 9.]));

    let res = ndarray::stack(Axis(1), &[a.view(), c.view()]);
    assert_eq!(res.unwrap_err().kind(), ErrorKind::IncompatibleShape);

    let res = ndarray::stack(Axis(2), &[a.view(), c.view()]);
    assert_eq!(res.unwrap_err().kind(), ErrorKind::OutOfBounds);

    let res: Result<Array2<f64>, _> = ndarray::stack(Axis(0), &[]);
    assert_eq!(res.unwrap_err().kind(), ErrorKind::Unsupported);
}

#[test]
fn stacking() {
    let a = arr2(&[[2., 2.], [3., 3.]]);
    let b = ndarray::stack_new_axis(Axis(0), vec![a.view(), a.view()]).unwrap();
    assert_eq!(b, arr3(&[[[2., 2.], [3., 3.]], [[2., 2.], [3., 3.]]]));

    let c = arr2(&[[3., 2., 3.], [2., 3., 2.]]);
    let res = ndarray::stack_new_axis(Axis(1), vec![a.view(), c.view()]);
    assert_eq!(res.unwrap_err().kind(), ErrorKind::IncompatibleShape);

    let res = ndarray::stack_new_axis(Axis(3), vec![a.view(), a.view()]);
    assert_eq!(res.unwrap_err().kind(), ErrorKind::OutOfBounds);

    let res: Result<Array2<f64>, _> = ndarray::stack_new_axis::<_, Ix1>(Axis(0), vec![]);
    assert_eq!(res.unwrap_err().kind(), ErrorKind::Unsupported);
}
