
#[macro_use(stack)]
extern crate ndarray;


use ndarray::{
    arr2,
    Axis,
};

#[test]
fn vstack() {
    let a = arr2(&[[2., 2.],
                   [3., 3.]]);
    let b = ndarray::stack(&[a.view(), a.view()], Axis(0)).unwrap();
    assert_eq!(b, arr2(&[[2., 2.],
                         [3., 3.],
                         [2., 2.],
                         [3., 3.]]));

    let b = stack!(Axis(1), a.view(), a);
}
