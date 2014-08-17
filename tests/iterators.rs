extern crate ndarray;

use ndarray::Array;
use ndarray::Ix;

#[test]
fn double_ended()
{
    let a = Array::from_iter(range(0.0, 8.0f32));
    let mut it = a.iter().map(|x| *x);
    assert_eq!(it.next(), Some(0.));
    assert_eq!(it.next_back(), Some(7.));
    assert_eq!(it.next(), Some(1.));
    assert_eq!(it.rev().last(), Some(2.));
}

#[test]
fn indexed()
{
    let a = Array::from_iter(range(0.0, 8.0f32));
    for (i, elt) in a.indexed_iter() {
        assert_eq!(i, *elt as Ix);
    }
    let a = a.reshape((2u, 4u, 1u));
    let (mut i, mut j, k) = (0u, 0u, 0u);
    for (idx, elt) in a.indexed_iter() {
        assert_eq!(idx, (i, j, k));
        j += 1;
        if j == 4 {
            j = 0;
            i += 1;
        }
        println!("{}", (idx, elt));
    }
}
