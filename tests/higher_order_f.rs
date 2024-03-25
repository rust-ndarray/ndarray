use ndarray::prelude::*;

#[test]
#[should_panic]
fn test_fold_axis_oob() {
    let a = arr2(&[[1., 2.], [3., 4.]]);
    a.fold_axis(Axis(2), 0., |x, y| x + y);
}
