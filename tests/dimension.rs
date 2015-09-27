extern crate ndarray;

use ndarray::{
    arr0,
    Array,
    //Dimension,
    RemoveAxis,
};

#[test]
fn remove_axis()
{
    assert_eq!(3.remove_axis(0), ());
    assert_eq!((1, 2).remove_axis(0), 2);
    assert_eq!((4, 5, 6).remove_axis(1), (4, 6));

    assert_eq!(vec![1,2].remove_axis(0), vec![2]);
    assert_eq!(vec![4, 5, 6].remove_axis(1), vec![4, 6]);

    let a = Array::<f32, _>::zeros(vec![4,5,6]);
    let b = a.subview(1, 0).reshape_into((4, 6)).reshape_into(vec![2, 3, 4]);
}
