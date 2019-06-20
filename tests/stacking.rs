extern crate ndarray;

use ndarray::{arr2, aview1, stack, Array2, Axis};
use ndarray::{ShapeError, ShapeErrorKind};

#[test]
fn stacking() {
    let a = arr2(&[[2., 2.], [3., 3.]]);
    let b = ndarray::stack(Axis(0), &[a.view(), a.view()]).unwrap();
    assert_eq!(b, arr2(&[[2., 2.], [3., 3.], [2., 2.], [3., 3.]]));

    let c = stack![Axis(0), a, b];
    assert_eq!(
        c,
        arr2(&[[2., 2.], [3., 3.], [2., 2.], [3., 3.], [2., 2.], [3., 3.]])
    );

    let d = stack![Axis(0), a.row(0), &[9., 9.]];
    assert_eq!(d, aview1(&[2., 2., 9., 9.]));

    let res = ndarray::stack(Axis(1), &[a.view(), c.view()]);
    assert_eq!(
        res.unwrap_err(),
        ShapeError::from(ShapeErrorKind::IncompatibleShape {
            message: format!(
                "Difference between raw dimension: {:?} and common dimension: {:?}, \
                 apart from along `axis`, array: {:?}.",
                [6],
                [2],
                1
            )
        })
    );

    let res = ndarray::stack(Axis(2), &[a.view(), c.view()]);
    assert_eq!(
        res.unwrap_err(),
        ShapeError::from(ShapeErrorKind::OutOfBounds {
            message: format!(
                "The axis index: {:?} greater than the number of raw dimensions: {:?}.",
                2, 2
            )
        })
    );

    let res: Result<Array2<f64>, _> = ndarray::stack(Axis(0), &[]);
    assert_eq!(
        res.unwrap_err(),
        ShapeError::from(ShapeErrorKind::Unsupported {
            message: String::from("Stack `arrays` param is empty.")
        })
    );
}
