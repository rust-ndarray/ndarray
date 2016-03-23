
extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;

use rand::distributions::Range;
use ndarray::OwnedArray;
use ndarray_rand::RandomExt;

#[test]
fn test_dim() {
    let (mm, nn) = (5, 5);
    for m in 0..mm {
        for n in 0..nn {
            let a = OwnedArray::random((m, n), Range::new(0., 2.));
            assert_eq!(a.shape(), &[m, n]);
            assert!(a.iter().all(|x| *x <= 2.));
            assert!(a.iter().all(|x| *x >= 0.));
        }
    }
}
