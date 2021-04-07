
use ndarray::prelude::*;
use ndarray::{ShapeError, ErrorKind};

#[test]
fn append_row() {
    let mut a = Array::zeros((0, 4));
    a.append_row(aview1(&[0., 1., 2., 3.])).unwrap();
    a.append_row(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a.shape(), &[2, 4]);

    assert_eq!(a,
        array![[0., 1., 2., 3.],
               [4., 5., 6., 7.]]);

    assert_eq!(a.append_row(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.append_column(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.append_column(aview1(&[1., 2.])),
        Ok(()));
    assert_eq!(a,
        array![[0., 1., 2., 3., 1.],
               [4., 5., 6., 7., 2.]]);
}

#[test]
fn append_row_wrong_layout() {
    let mut a = Array::zeros((0, 4));
    a.append_row(aview1(&[0., 1., 2., 3.])).unwrap();
    a.append_row(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a.shape(), &[2, 4]);

    //assert_eq!(a.append_column(aview1(&[1., 2.])), Err(ShapeError::from_kind(ErrorKind::IncompatibleLayout)));

    assert_eq!(a,
        array![[0., 1., 2., 3.],
               [4., 5., 6., 7.]]);

    // Clone the array

    let mut dim = a.raw_dim();
    dim[1] = 0;
    let mut b = Array::zeros(dim);
    b.append(Axis(1), a.view()).unwrap();
    assert_eq!(b.append_column(aview1(&[1., 2.])), Ok(()));
    assert_eq!(b,
        array![[0., 1., 2., 3., 1.],
               [4., 5., 6., 7., 2.]]);
}

#[test]
fn append_row_error() {
    let mut a = Array::zeros((3, 4));

    assert_eq!(a.append_row(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.append_column(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.append_column(aview1(&[1., 2., 3.])),
        Ok(()));
    assert_eq!(a.t(),
        array![[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [1., 2., 3.]]);
}

#[test]
fn append_row_existing() {
    let mut a = Array::zeros((1, 4));
    a.append_row(aview1(&[0., 1., 2., 3.])).unwrap();
    a.append_row(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a.shape(), &[3, 4]);

    assert_eq!(a,
        array![[0., 0., 0., 0.],
               [0., 1., 2., 3.],
               [4., 5., 6., 7.]]);

    assert_eq!(a.append_row(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.append_column(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.append_column(aview1(&[1., 2., 3.])),
        Ok(()));
    assert_eq!(a,
        array![[0., 0., 0., 0., 1.],
               [0., 1., 2., 3., 2.],
               [4., 5., 6., 7., 3.]]);
}

#[test]
fn append_row_col_len_1() {
    // Test appending 1 row and then cols from shape 1 x 1
    let mut a = Array::zeros((1, 1));
    a.append_row(aview1(&[1.])).unwrap(); // shape 2 x 1
    a.append_column(aview1(&[2., 3.])).unwrap(); // shape 2 x 2
    assert_eq!(a.append_row(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    //assert_eq!(a.append_row(aview1(&[1., 2.])), Err(ShapeError::from_kind(ErrorKind::IncompatibleLayout)));
    a.append_column(aview1(&[4., 5.])).unwrap(); // shape 2 x 3
    assert_eq!(a.shape(), &[2, 3]);

    assert_eq!(a,
        array![[0., 2., 4.],
               [1., 3., 5.]]);
}

#[test]
fn append_column() {
    let mut a = Array::zeros((4, 0));
    a.append_column(aview1(&[0., 1., 2., 3.])).unwrap();
    a.append_column(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a.shape(), &[4, 2]);

    assert_eq!(a.t(),
        array![[0., 1., 2., 3.],
               [4., 5., 6., 7.]]);
}

#[test]
fn append_array1() {
    let mut a = Array::zeros((0, 4));
    a.append(Axis(0), aview2(&[[0., 1., 2., 3.]])).unwrap();
    println!("{:?}", a);
    a.append(Axis(0), aview2(&[[4., 5., 6., 7.]])).unwrap();
    println!("{:?}", a);
    //a.append_column(aview1(&[4., 5., 6., 7.])).unwrap();
    //assert_eq!(a.shape(), &[4, 2]);

    assert_eq!(a,
        array![[0., 1., 2., 3.],
               [4., 5., 6., 7.]]);

    a.append(Axis(0), aview2(&[[5., 5., 4., 4.], [3., 3., 2., 2.]])).unwrap();
    println!("{:?}", a);
    assert_eq!(a,
        array![[0., 1., 2., 3.],
               [4., 5., 6., 7.],
               [5., 5., 4., 4.],
               [3., 3., 2., 2.]]);
}

#[test]
fn append_array_3d() {
    let mut a = Array::zeros((0, 2, 2));
    a.append(Axis(0), array![[[0, 1], [2, 3]]].view()).unwrap();
    println!("{:?}", a);

    let aa = array![[[51, 52], [53, 54]], [[55, 56], [57, 58]]];
    let av = aa.view();
    println!("Send {:?} to append", av);
    a.append(Axis(0), av.clone()).unwrap();

    a.swap_axes(0, 1);
    let aa = array![[[71, 72], [73, 74]], [[75, 76], [77, 78]]];
    let mut av = aa.view();
    av.swap_axes(0, 1);
    println!("Send {:?} to append", av);
    a.append(Axis(1), av.clone()).unwrap();
    println!("{:?}", a);
    let aa = array![[[81, 82], [83, 84]], [[85, 86], [87, 88]]];
    let mut av = aa.view();
    av.swap_axes(0, 1);
    println!("Send {:?} to append", av);
    a.append(Axis(1), av).unwrap();
    println!("{:?}", a);
    assert_eq!(a,
        array![[[0, 1],
                [51, 52],
                [55, 56],
                [71, 72],
                [75, 76],
                [81, 82],
                [85, 86]],
               [[2, 3],
                [53, 54],
                [57, 58],
                [73, 74],
                [77, 78],
                [83, 84],
                [87, 88]]]);
}

#[test]
fn test_append_2d() {
    // create an empty array and append
    let mut a = Array::zeros((0, 4));
    let ones = ArrayView::from(&[1.; 12]).into_shape((3, 4)).unwrap();
    let zeros = ArrayView::from(&[0.; 8]).into_shape((2, 4)).unwrap();
    a.append(Axis(0), ones).unwrap();
    a.append(Axis(0), zeros).unwrap();
    a.append(Axis(0), ones).unwrap();
    println!("{:?}", a);
    assert_eq!(a.shape(), &[8, 4]);
    for (i, row) in a.rows().into_iter().enumerate() {
        let ones = i < 3 || i >= 5;
        assert!(row.iter().all(|&x| x == ones as i32 as f64), "failed on lane {}", i);
    }

    let mut a = Array::zeros((0, 4));
    a = a.reversed_axes();
    let ones = ones.reversed_axes();
    let zeros = zeros.reversed_axes();
    a.append(Axis(1), ones).unwrap();
    a.append(Axis(1), zeros).unwrap();
    a.append(Axis(1), ones).unwrap();
    println!("{:?}", a);
    assert_eq!(a.shape(), &[4, 8]);

    for (i, row) in a.columns().into_iter().enumerate() {
        let ones = i < 3 || i >= 5;
        assert!(row.iter().all(|&x| x == ones as i32 as f64), "failed on lane {}", i);
    }
}

#[test]
fn test_append_middle_axis() {
    // ensure we can append to Axis(1) by letting it become outermost
    let mut a = Array::<i32, _>::zeros((3, 0, 2));
    a.append(Axis(1), Array::from_iter(0..12).into_shape((3, 2, 2)).unwrap().view()).unwrap();
    println!("{:?}", a);
    a.append(Axis(1), Array::from_iter(12..24).into_shape((3, 2, 2)).unwrap().view()).unwrap();
    println!("{:?}", a);

    // ensure we can append to Axis(1) by letting it become outermost
    let mut a = Array::<i32, _>::zeros((3, 1, 2));
    a.append(Axis(1), Array::from_iter(0..12).into_shape((3, 2, 2)).unwrap().view()).unwrap();
    println!("{:?}", a);
    a.append(Axis(1), Array::from_iter(12..24).into_shape((3, 2, 2)).unwrap().view()).unwrap();
    println!("{:?}", a);
}

#[test]
fn test_append_zero_size() {
    {
        let mut a = Array::<i32, _>::zeros((0, 0));
        a.append(Axis(0), aview2(&[[]])).unwrap();
        a.append(Axis(0), aview2(&[[]])).unwrap();
        assert_eq!(a.len(), 0);
        assert_eq!(a.shape(), &[2, 0]);
    }

    {
        let mut a = Array::<i32, _>::zeros((0, 0));
        a.append(Axis(1), ArrayView::from(&[]).into_shape((0, 1)).unwrap()).unwrap();
        a.append(Axis(1), ArrayView::from(&[]).into_shape((0, 1)).unwrap()).unwrap();
        assert_eq!(a.len(), 0);
        assert_eq!(a.shape(), &[0, 2]);
    }
}
