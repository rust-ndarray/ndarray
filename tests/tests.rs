extern crate ndarray;

use ndarray::Array;

#[test]
fn char_array()
{
    // test compilation & basics of non-numerical array
    let cc = Array::from_iter("alphabet".chars()).reshape_into((4, 2));
    assert!(cc.subview(1, 0) == Array::from_iter("apae".chars()).view());
}
