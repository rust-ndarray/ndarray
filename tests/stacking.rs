
extern crate ndarray;


use ndarray::{
    arr2,
    Axis,
};

use ndarray::stacking::ArrayStackingExt;

#[test]
fn vstack() {
    let a = arr2(&[[2., 2.],
                   [3., 3.]]);
    let b = [a.view(), a.view()].stack(Axis(0));
    assert_eq!(b, arr2(&[[2., 2.],
                         [3., 3.],
                         [2., 2.],
                         [3., 3.]]));
}
