
use ndarray::prelude::*;
use ndarray::{ShapeError, ErrorKind};

#[test]
fn push_row() {
    let mut a = Array::zeros((0, 4));
    a.push_row(aview1(&[0., 1., 2., 3.])).unwrap();
    a.push_row(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a.shape(), &[2, 4]);

    assert_eq!(a,
        array![[0., 1., 2., 3.],
               [4., 5., 6., 7.]]);

    assert_eq!(a.push_row(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.push_column(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.push_column(aview1(&[1., 2.])),
        Ok(()));
    assert_eq!(a,
        array![[0., 1., 2., 3., 1.],
               [4., 5., 6., 7., 2.]]);
}

#[test]
fn push_row_wrong_layout() {
    let mut a = Array::zeros((0, 4));
    a.push_row(aview1(&[0., 1., 2., 3.])).unwrap();
    a.push_row(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a.shape(), &[2, 4]);

    assert_eq!(a,
        array![[0., 1., 2., 3.],
               [4., 5., 6., 7.]]);
    assert_eq!(a.strides(), &[4, 1]);

    // Changing the memory layout to fit the next append
    let mut a2 = a.clone();
    a2.push_column(aview1(&[1., 2.])).unwrap();
    assert_eq!(a2,
        array![[0., 1., 2., 3., 1.],
               [4., 5., 6., 7., 2.]]);
    assert_eq!(a2.strides(), &[1, 2]);


    // Clone the array

    let mut dim = a.raw_dim();
    dim[1] = 0;
    let mut b = Array::zeros(dim);
    b.append(Axis(1), a.view()).unwrap();
    assert_eq!(b.push_column(aview1(&[1., 2.])), Ok(()));
    assert_eq!(b,
        array![[0., 1., 2., 3., 1.],
               [4., 5., 6., 7., 2.]]);
}

#[test]
fn push_row_neg_stride_1() {
    let mut a = Array::zeros((0, 4));
    a.push_row(aview1(&[0., 1., 2., 3.])).unwrap();
    a.push_row(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a.shape(), &[2, 4]);

    assert_eq!(a,
        array![[0., 1., 2., 3.],
               [4., 5., 6., 7.]]);
    assert_eq!(a.strides(), &[4, 1]);

    a.invert_axis(Axis(0));

    // Changing the memory layout to fit the next append
    let mut a2 = a.clone();
    println!("a = {:?}", a);
    println!("a2 = {:?}", a2);
    a2.push_column(aview1(&[1., 2.])).unwrap();
    assert_eq!(a2,
        array![[4., 5., 6., 7., 1.],
               [0., 1., 2., 3., 2.]]);
    assert_eq!(a2.strides(), &[1, 2]);

    a.invert_axis(Axis(1));
    let mut a3 = a.clone();
    a3.push_row(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a3,
        array![[7., 6., 5., 4.],
               [3., 2., 1., 0.],
               [4., 5., 6., 7.]]);
    assert_eq!(a3.strides(), &[4, 1]);

    a.invert_axis(Axis(0));
    let mut a4 = a.clone();
    a4.push_row(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a4,
        array![[3., 2., 1., 0.],
               [7., 6., 5., 4.],
               [4., 5., 6., 7.]]);
    assert_eq!(a4.strides(), &[4, -1]);
}

#[test]
fn push_row_neg_stride_2() {
    let mut a = Array::zeros((0, 4));
    a.push_row(aview1(&[0., 1., 2., 3.])).unwrap();
    a.push_row(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a.shape(), &[2, 4]);

    assert_eq!(a,
        array![[0., 1., 2., 3.],
               [4., 5., 6., 7.]]);
    assert_eq!(a.strides(), &[4, 1]);

    a.invert_axis(Axis(1));

    // Changing the memory layout to fit the next append
    let mut a2 = a.clone();
    println!("a = {:?}", a);
    println!("a2 = {:?}", a2);
    a2.push_column(aview1(&[1., 2.])).unwrap();
    assert_eq!(a2,
        array![[3., 2., 1., 0., 1.],
               [7., 6., 5., 4., 2.]]);
    assert_eq!(a2.strides(), &[1, 2]);

    a.invert_axis(Axis(0));
    let mut a3 = a.clone();
    a3.push_row(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a3,
        array![[7., 6., 5., 4.],
               [3., 2., 1., 0.],
               [4., 5., 6., 7.]]);
    assert_eq!(a3.strides(), &[4, 1]);

    a.invert_axis(Axis(1));
    let mut a4 = a.clone();
    a4.push_row(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a4,
        array![[4., 5., 6., 7.],
               [0., 1., 2., 3.],
               [4., 5., 6., 7.]]);
    assert_eq!(a4.strides(), &[4, 1]);
}

#[test]
fn push_row_error() {
    let mut a = Array::zeros((3, 4));

    assert_eq!(a.push_row(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.push_column(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.push_column(aview1(&[1., 2., 3.])),
        Ok(()));
    assert_eq!(a.t(),
        array![[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [1., 2., 3.]]);
}

#[test]
fn push_row_existing() {
    let mut a = Array::zeros((1, 4));
    a.push_row(aview1(&[0., 1., 2., 3.])).unwrap();
    a.push_row(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a.shape(), &[3, 4]);

    assert_eq!(a,
        array![[0., 0., 0., 0.],
               [0., 1., 2., 3.],
               [4., 5., 6., 7.]]);

    assert_eq!(a.push_row(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.push_column(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    assert_eq!(a.push_column(aview1(&[1., 2., 3.])),
        Ok(()));
    assert_eq!(a,
        array![[0., 0., 0., 0., 1.],
               [0., 1., 2., 3., 2.],
               [4., 5., 6., 7., 3.]]);
}

#[test]
fn push_row_col_len_1() {
    // Test appending 1 row and then cols from shape 1 x 1
    let mut a = Array::zeros((1, 1));
    a.push_row(aview1(&[1.])).unwrap(); // shape 2 x 1
    a.push_column(aview1(&[2., 3.])).unwrap(); // shape 2 x 2
    assert_eq!(a.push_row(aview1(&[1.])),
        Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    //assert_eq!(a.push_row(aview1(&[1., 2.])), Err(ShapeError::from_kind(ErrorKind::IncompatibleLayout)));
    a.push_column(aview1(&[4., 5.])).unwrap(); // shape 2 x 3
    assert_eq!(a.shape(), &[2, 3]);

    assert_eq!(a,
        array![[0., 2., 4.],
               [1., 3., 5.]]);
}

#[test]
fn push_column() {
    let mut a = Array::zeros((4, 0));
    a.push_column(aview1(&[0., 1., 2., 3.])).unwrap();
    a.push_column(aview1(&[4., 5., 6., 7.])).unwrap();
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
    //a.push_column(aview1(&[4., 5., 6., 7.])).unwrap();
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

#[test]
fn push_row_neg_stride_3() {
    let mut a = Array::zeros((0, 4));
    a.push_row(aview1(&[0., 1., 2., 3.])).unwrap();
    a.invert_axis(Axis(1));
    a.push_row(aview1(&[4., 5., 6., 7.])).unwrap();
    assert_eq!(a.shape(), &[2, 4]);
    assert_eq!(a, array![[3., 2., 1., 0.], [4., 5., 6., 7.]]);
    assert_eq!(a.strides(), &[4, -1]);
}

#[test]
fn push_row_ignore_strides_length_one_axes() {
    let strides = &[0, 1, 10, 20];
    for invert in &[vec![], vec![0], vec![1], vec![0, 1]] {
        for &stride0 in strides {
            for &stride1 in strides {
                let mut a =
                    Array::from_shape_vec([1, 1].strides([stride0, stride1]), vec![0.]).unwrap();
                for &ax in invert {
                    a.invert_axis(Axis(ax));
                }
                a.push_row(aview1(&[1.])).unwrap();
                assert_eq!(a.shape(), &[2, 1]);
                assert_eq!(a, array![[0.], [1.]]);
                assert_eq!(a.stride_of(Axis(0)), 1);
            }
        }
    }
}

#[test]
#[should_panic(expected = "IncompatibleShape")]
fn zero_dimensional_error1() {
    let mut a = Array::zeros(()).into_dyn();
    a.append(Axis(0), arr0(0).into_dyn().view()).unwrap();
}

#[test]
#[should_panic(expected = "IncompatibleShape")]
fn zero_dimensional_error2() {
    let mut a = Array::zeros(()).into_dyn();
    a.push(Axis(0), arr0(0).into_dyn().view()).unwrap();
}

#[test]
fn zero_dimensional_ok() {
    let mut a = Array::zeros(0);
    let one = aview0(&1);
    let two = aview0(&2);
    a.push(Axis(0), two).unwrap();
    a.push(Axis(0), one).unwrap();
    a.push(Axis(0), one).unwrap();
    assert_eq!(a, array![2, 1, 1]);
}
