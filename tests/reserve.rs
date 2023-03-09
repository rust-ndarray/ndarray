use ndarray::prelude::*;

#[test]
fn reserve_1d()
{
    let mut a = Array1::<i32>::zeros((4,));
    a.reserve(Axis(0), 1000);
    assert_eq!(a.shape(), &[4]);
    assert!(a.into_raw_vec().capacity() >= 1004);
}

#[test]
fn reserve_3d()
{
    let mut a = Array3::<i32>::zeros((0, 4, 8));
    a.reserve(Axis(0), 10);
    assert_eq!(a.shape(), &[0, 4, 8]);
    assert!(a.into_raw_vec().capacity() >= 4 * 8 * 10);
}

#[test]
fn reserve_empty_3d()
{
    let mut a = Array3::<i32>::zeros((0, 0, 0));
    a.reserve(Axis(0), 10);
}

#[test]
fn reserve_3d_axis1()
{
    let mut a = Array3::<i32>::zeros((2, 4, 8));
    a.reserve(Axis(1), 10);
    assert!(a.into_raw_vec().capacity() >= 2 * 8 * 10);
}

#[test]
fn reserve_3d_repeat()
{
    let mut a = Array3::<i32>::zeros((2, 4, 8));
    a.reserve(Axis(1), 10);
    a.reserve(Axis(2), 30);
    assert!(a.into_raw_vec().capacity() >= 2 * 4 * 30);
}

#[test]
fn reserve_2d_with_data()
{
    let mut a = array![[1, 2], [3, 4], [5, 6]];
    a.reserve(Axis(1), 100);
    assert_eq!(a, array![[1, 2], [3, 4], [5, 6]]);
    assert!(a.into_raw_vec().capacity() >= 3 * 100);
}
