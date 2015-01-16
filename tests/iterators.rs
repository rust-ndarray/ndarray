extern crate ndarray;

use ndarray::Array;
use ndarray::{Ix, Si, S, arr2};

#[test]
fn double_ended()
{
    let a = Array::range(0.0, 8.0f32);
    let mut it = a.iter().map(|x| *x);
    assert_eq!(it.next(), Some(0.));
    assert_eq!(it.next_back(), Some(7.));
    assert_eq!(it.next(), Some(1.));
    assert_eq!(it.rev().last(), Some(2.));
}

#[test]
fn indexed()
{
    let a = Array::range(0.0, 8.0f32);
    for (i, elt) in a.indexed_iter() {
        assert_eq!(i, *elt as Ix);
    }
    let a = a.reshape((2, 4, 1));
    let (mut i, mut j, k) = (0, 0, 0);
    for (idx, elt) in a.indexed_iter() {
        assert_eq!(idx, (i, j, k));
        j += 1;
        if j == 4 {
            j = 0;
            i += 1;
        }
        println!("{:?}", (idx, elt));
    }
}

#[test]
fn indexed2()
{
    let a = Array::range(0.0, 8.0f32);
    let mut iter = a.iter();
    iter.next();
    let mut iter = iter.indexed();
    for (i, elt) in iter {
        assert_eq!(i, *elt as Ix);
    }
    let a = a.reshape((2, 4, 1));
    let (mut i, mut j, k) = (0, 0, 0);
    for (idx, elt) in a.iter().indexed() {
        assert_eq!(idx, (i, j, k));
        j += 1;
        if j == 4 {
            j = 0;
            i += 1;
        }
        println!("{:?}", (idx, elt));
    }
}

#[test]
fn indexed3()
{
    let a = Array::range(0.0, 8.0f32);
    let mut a = a.reshape((2, 4, 1));
    let (mut i, mut j, k) = (0, 0, 0);
    for (idx, elt) in a.slice_iter_mut(&[S, Si(1, None, 2), S]).indexed()
    {
        assert_eq!(idx, (i, j, k));
        j += 1;
        if j == 2 {
            j = 0;
            i += 1;
        }
        *elt = -1.;
        println!("{:?}", (idx, elt));
    }
    let a = a.reshape((2, 4));
    assert_eq!( a, arr2(&[[0., -1., 2., -1.],
                          [4., -1., 6., -1.]]));
}
