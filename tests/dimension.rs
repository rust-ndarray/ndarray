extern crate ndarray;

use ndarray::{
    Array,
    RemoveAxis,
    arr2,
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
    let _b = a.into_subview(1, 0).reshape((4, 6)).reshape(vec![2, 3, 4]);
    
}

#[test]
fn dyn_dimension()
{
    let a = arr2(&[[1., 2.], [3., 4.0]]).reshape(vec![2, 2]);
    assert_eq!(&a - &a, Array::zeros(vec![2, 2]));
    assert_eq!(a[&[0, 0][..]], 1.);
    assert_eq!(a[vec![0, 0]], 1.);

    let mut dim = vec![1; 1024];
    dim[16] = 4;
    dim[17] = 3;
    let z = Array::<f32, _>::zeros(dim.clone());
    assert_eq!(z.shape(), &dim[..]);
}

