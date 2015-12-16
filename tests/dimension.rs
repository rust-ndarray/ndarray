extern crate ndarray;

use ndarray::{
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
    let _b = a.subview(1, 0).reshape((4, 6)).reshape(vec![2, 3, 4]);
    
}
