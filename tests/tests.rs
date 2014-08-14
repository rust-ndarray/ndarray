extern crate test;
extern crate ndarray;

use ndarray::Array;

#[test]
fn char_array()
{
    // test compilation & basics of non-numerical array
    let cc = Array::from_iter("alphabet".chars()).reshape((4u, 2u));
    assert!(cc.subview(1, 0) == Array::from_iter("apae".chars()));
}

#[bench]
fn time_matmul(b: &mut test::Bencher)
{
    b.iter(|| {
        let mut a: Array<uint, (uint, uint)> = Array::zeros((2u, 3u));
        for (i, elt) in a.iter_mut().enumerate() {
            *elt = i;
        }

        let mut b: Array<uint, (uint, uint)> = Array::zeros((3u, 4u));
        for (i, elt) in b.iter_mut().enumerate() {
            *elt = i;
        }

        let c = a.mat_mul(&b);
        unsafe {
            let result = Array::from_vec_dim((2u, 4u), vec![20u, 23, 26, 29, 56, 68, 80, 92]);
            assert!(c == result);
        }
    })
}
