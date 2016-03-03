extern crate ndarray;

use ndarray::{
    RcArray,
    OwnedArray,
    RemoveAxis,
    arr2,
    Axis,
    Ix,
    Dimension
};

#[test]
fn remove_axis()
{
    assert_eq!(3.remove_axis(0), ());
    assert_eq!((1, 2).remove_axis(0), 2);
    assert_eq!((4, 5, 6).remove_axis(1), (4, 6));

    assert_eq!(vec![1,2].remove_axis(0), vec![2]);
    assert_eq!(vec![4, 5, 6].remove_axis(1), vec![4, 6]);

    let a = RcArray::<f32, _>::zeros((4,5));
    a.subview(Axis(1), 0);

    let a = RcArray::<f32, _>::zeros(vec![4,5,6]);
    let _b = a.into_subview(Axis(1), 0).reshape((4, 6)).reshape(vec![2, 3, 4]);
    
}

#[test]
fn dyn_dimension()
{
    let a = arr2(&[[1., 2.], [3., 4.0]]).into_shape(vec![2, 2]).unwrap();
    assert_eq!(&a - &a, OwnedArray::zeros(vec![2, 2]));
    assert_eq!(a[&[0, 0][..]], 1.);
    assert_eq!(a[vec![0, 0]], 1.);

    let mut dim = vec![1; 1024];
    dim[16] = 4;
    dim[17] = 3;
    let z = OwnedArray::<f32, _>::zeros(dim.clone());
    assert_eq!(z.shape(), &dim[..]);
}

#[test]
fn index_axis()
{
    assert_eq!(3.index(Axis(0)), &3);
    assert_eq!((3, 2).index(Axis(1)), &2);

    let mut dim = (2, 3, 3);
    *dim.index_mut(Axis(2)) = 1;
    assert_eq!(dim.index(Axis(2)), &1);

    let a: OwnedArray<f64, (Ix, Ix, Ix)> = OwnedArray::zeros(dim);
    assert_eq!(a.dim().index(Axis(1)), &3);
}
