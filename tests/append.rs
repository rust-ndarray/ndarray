
use ndarray::prelude::*;
use ndarray::{ShapeError, ErrorKind};

#[test]
fn append_row() {
    let mut a = Array::zeros((0, 4));
    a.try_append_row(aview1(&[0., 1., 2., 3.])).unwrap();
    a.try_append_row(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a.shape(), &[2, 4]);

    assert_eq!(a,
        array![[0., 1., 2., 3.],
               [4., 5., 6., 7.]]);

    assert_eq!(a.try_append_row(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.try_append_column(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.try_append_column(aview1(&[1., 2.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleLayout)));
}

#[test]
fn append_row_error() {
    let mut a = Array::zeros((3, 4));

    assert_eq!(a.try_append_row(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.try_append_column(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.try_append_column(aview1(&[1., 2., 3.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleLayout)));
}

#[test]
fn append_row_existing() {
    let mut a = Array::zeros((1, 4));
    a.try_append_row(aview1(&[0., 1., 2., 3.])).unwrap();
    a.try_append_row(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a.shape(), &[3, 4]);

    assert_eq!(a,
        array![[0., 0., 0., 0.],
               [0., 1., 2., 3.],
               [4., 5., 6., 7.]]);

    assert_eq!(a.try_append_row(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.try_append_column(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.try_append_column(aview1(&[1., 2., 3.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleLayout)));
}

#[test]
fn append_row_col_len_1() {
    // Test appending 1 row and then cols from shape 1 x 1
    let mut a = Array::zeros((1, 1));
    a.try_append_row(aview1(&[1.])).unwrap(); // shape 2 x 1
    a.try_append_column(aview1(&[2., 3.])).unwrap(); // shape 2 x 2
    assert_eq!(a.try_append_row(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.try_append_row(aview1(&[1., 2.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleLayout)));
    a.try_append_column(aview1(&[4., 5.])).unwrap(); // shape 2 x 3
    assert_eq!(a.shape(), &[2, 3]);

    assert_eq!(a,
        array![[0., 2., 4.],
               [1., 3., 5.]]);
}

#[test]
fn append_column() {
    let mut a = Array::zeros((4, 0));
    a.try_append_column(aview1(&[0., 1., 2., 3.])).unwrap();
    a.try_append_column(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a.shape(), &[4, 2]);

    assert_eq!(a.t(),
        array![[0., 1., 2., 3.],
               [4., 5., 6., 7.]]);
}

#[test]
fn append_array1() {
    let mut a = Array::zeros((0, 4));
    a.try_append_array(Axis(0), aview2(&[[0., 1., 2., 3.]])).unwrap();
    println!("{:?}", a);
    a.try_append_array(Axis(0), aview2(&[[4., 5., 6., 7.]])).unwrap();
    println!("{:?}", a);
    //a.try_append_column(aview1(&[4., 5., 6., 7.])).unwrap();
    //assert_eq!(a.shape(), &[4, 2]);

    assert_eq!(a,
        array![[0., 1., 2., 3.],
               [4., 5., 6., 7.]]);

    a.try_append_array(Axis(0), aview2(&[[5., 5., 4., 4.], [3., 3., 2., 2.]])).unwrap();
    println!("{:?}", a);
    assert_eq!(a,
        array![[0., 1., 2., 3.],
               [4., 5., 6., 7.],
               [5., 5., 4., 4.],
               [3., 3., 2., 2.]]);
}
