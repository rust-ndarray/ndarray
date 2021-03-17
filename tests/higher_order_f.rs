use ndarray::prelude::*;

#[test]
#[should_panic]
fn test_fold_axis_oob() {
    let a = arr2(&[[1., 2.], [3., 4.]]);
    a.fold_axis(Axis(2), 0., |x, y| x + y);
}

#[test]
fn assign() {
    let mut a = arr2(&[[1., 2.], [3., 4.]]);
    let b = arr2(&[[1., 3.], [2., 4.]]);
    a.assign(&b);
    assert_eq!(a, b);

    /* Test broadcasting */
    a.assign(&ArcArray::zeros(1));
    assert_eq!(a, ArcArray::zeros((2, 2)));

    /* Test other type */
    a.assign(&Array::from_elem((2, 2), 3.));
    assert_eq!(a, ArcArray::from_elem((2, 2), 3.));

    /* Test mut view */
    let mut a = arr2(&[[1, 2], [3, 4]]);
    {
        let mut v = a.view_mut();
        v.slice_collapse(s![..1, ..]);
        v.fill(0);
    }
    assert_eq!(a, arr2(&[[0, 0], [3, 4]]));
}


#[test]
fn assign_to() {
    let mut a = arr2(&[[1., 2.], [3., 4.]]);
    let b = arr2(&[[0., 3.], [2., 0.]]);
    b.assign_to(&mut a);
    assert_eq!(a, b);
}
