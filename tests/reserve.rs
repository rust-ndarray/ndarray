use ndarray::prelude::*;

fn into_raw_vec_capacity<T, D: Dimension>(a: Array<T, D>) -> usize
{
    a.into_raw_vec_and_offset().0.capacity()
}

#[test]
fn reserve_1d()
{
    let mut a = Array1::<i32>::zeros((4,));
    a.reserve(Axis(0), 1000).unwrap();
    assert_eq!(a.shape(), &[4]);
    assert!(into_raw_vec_capacity(a) >= 1004);
}

#[test]
fn reserve_3d()
{
    let mut a = Array3::<i32>::zeros((0, 4, 8));
    a.reserve(Axis(0), 10).unwrap();
    assert_eq!(a.shape(), &[0, 4, 8]);
    assert!(into_raw_vec_capacity(a) >= 4 * 8 * 10);
}

#[test]
fn reserve_empty_3d()
{
    let mut a = Array3::<i32>::zeros((0, 0, 0));
    a.reserve(Axis(0), 10).unwrap();
}

#[test]
fn reserve_3d_axis1()
{
    let mut a = Array3::<i32>::zeros((2, 4, 8));
    a.reserve(Axis(1), 10).unwrap();
    assert!(into_raw_vec_capacity(a) >= 2 * 8 * 10);
}

#[test]
fn reserve_3d_repeat()
{
    let mut a = Array3::<i32>::zeros((2, 4, 8));
    a.reserve(Axis(1), 10).unwrap();
    a.reserve(Axis(2), 30).unwrap();
    assert!(into_raw_vec_capacity(a) >= 2 * 4 * 30);
}

#[test]
fn reserve_2d_with_data()
{
    let mut a = array![[1, 2], [3, 4], [5, 6]];
    a.reserve(Axis(1), 100).unwrap();
    assert_eq!(a, array![[1, 2], [3, 4], [5, 6]]);
    assert!(into_raw_vec_capacity(a) >= 3 * 100);
}

#[test]
fn reserve_2d_inverted_with_data()
{
    let mut a = array![[1, 2], [3, 4], [5, 6]];
    a.invert_axis(Axis(1));
    assert_eq!(a, array![[2, 1], [4, 3], [6, 5]]);
    a.reserve(Axis(1), 100).unwrap();
    assert_eq!(a, array![[2, 1], [4, 3], [6, 5]]);
    let (raw_vec, offset) = a.into_raw_vec_and_offset();
    assert!(raw_vec.capacity() >= 3 * 100);
    assert_eq!(offset, Some(1));
}
