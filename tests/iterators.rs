extern crate ndarray;

use ndarray::Array;

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
