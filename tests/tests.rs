extern crate test;
extern crate ndarray;

use ndarray::Array;
use ndarray::d2;

#[test]
fn char_array()
{
    // test compilation & basics of non-numerical array
    let cc = Array::from_iter("alphabet".chars()).reshape(d2(4, 2));
    assert!(cc.subview(1, 0) == Array::from_iter("apae".chars()));
}

#[bench]
fn time_matmul(b: &mut test::Bencher)
{
    b.iter(|| {
        let mut a: Array<uint, _> = Array::zeros(d2(2, 3));
        for (i, elt) in a.iter_mut().enumerate() {
            *elt = i;
        }

        let mut b: Array<uint, _> = Array::zeros(d2(3, 4));
        for (i, elt) in b.iter_mut().enumerate() {
            *elt = i;
        }

        let c = a.mat_mul(&b);
        unsafe {
            let result = Array::from_vec_dim(d2(2, 4), vec![20u, 23, 26, 29, 56, 68, 80, 92]);
            assert!(c == result);
        }
    })
}
