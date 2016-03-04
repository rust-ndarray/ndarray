
#[macro_use(stack)]
extern crate ndarray;


use ndarray::{
    arr2,
    Axis,
    Ix,
    OwnedArray,
    ErrorKind,
};

#[test]
fn stacking() {
    let a = arr2(&[[2., 2.],
                   [3., 3.]]);
    let b = ndarray::stack(Axis(0), &[a.view(), a.view()]).unwrap();
    assert_eq!(b, arr2(&[[2., 2.],
                         [3., 3.],
                         [2., 2.],
                         [3., 3.]]));

    let c = stack!(Axis(0), a.view(), b);
    assert_eq!(c, arr2(&[[2., 2.],
                         [3., 3.],
                         [2., 2.],
                         [3., 3.],
                         [2., 2.],
                         [3., 3.]]));

    let res = ndarray::stack(Axis(1), &[a.view(), c.view()]);
    assert_eq!(res.unwrap_err().kind(), ErrorKind::IncompatibleShape);

    let res = ndarray::stack(Axis(2), &[a.view(), c.view()]);
    assert_eq!(res.unwrap_err().kind(), ErrorKind::OutOfBounds);

    let res: Result<OwnedArray<f64, (Ix, Ix)>, _> = ndarray::stack(Axis(0), &[]);
    assert_eq!(res.unwrap_err().kind(), ErrorKind::Unsupported);
}
