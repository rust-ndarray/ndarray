extern crate ndarray;

use ndarray::{
    RcArray,
    Array,
    RemoveAxis,
    arr2,
    Axis,
    Dimension,
};

#[test]
fn remove_axis()
{
    assert_eq!(3.remove_axis(Axis(0)), ());
    assert_eq!((1, 2).remove_axis(Axis(0)), 2);
    assert_eq!((4, 5, 6).remove_axis(Axis(1)), (4, 6));

    assert_eq!(vec![1,2].remove_axis(Axis(0)), vec![2]);
    assert_eq!(vec![4, 5, 6].remove_axis(Axis(1)), vec![4, 6]);

    let a = RcArray::<f32, _>::zeros((4,5));
    a.subview(Axis(1), 0);

    let a = RcArray::<f32, _>::zeros(vec![4,5,6]);
    let _b = a.into_subview(Axis(1), 0).reshape((4, 6)).reshape(vec![2, 3, 4]);
    
}

#[test]
fn dyn_dimension()
{
    let a = arr2(&[[1., 2.], [3., 4.0]]).into_shape(vec![2, 2]).unwrap();
    assert_eq!(&a - &a, Array::zeros(vec![2, 2]));
    assert_eq!(a[&[0, 0][..]], 1.);
    assert_eq!(a[vec![0, 0]], 1.);

    let mut dim = vec![1; 1024];
    dim[16] = 4;
    dim[17] = 3;
    let z = Array::<f32, _>::zeros(dim.clone());
    assert_eq!(z.shape(), &dim[..]);
}

#[test]
fn fastest_varying_order() {
    let strides = (2, 8, 4, 1);
    let order = strides._fastest_varying_stride_order();
    assert_eq!(order.slice(), &[3, 0, 2, 1]);

    assert_eq!((1, 3)._fastest_varying_stride_order(), (0, 1));
    assert_eq!((7, 2)._fastest_varying_stride_order(), (1, 0));
    assert_eq!((6, 1, 3)._fastest_varying_stride_order(), (1, 2, 0));

    // it's important that it produces distinct indices. Prefer the stable order
    // where 0 is before 1 when they are equal.
    assert_eq!((2, 2)._fastest_varying_stride_order(), (0, 1));
    assert_eq!((2, 2, 1)._fastest_varying_stride_order(), (2, 0, 1));
    assert_eq!((2, 2, 3, 1, 2)._fastest_varying_stride_order(), (3, 0, 1, 4, 2));
}

