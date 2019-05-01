extern crate ndarray;

use ndarray::{s, Array};

#[test]
fn test_s()
{
    let a = Array::<usize, _>::zeros([3, 4]);
    let vi = a.slice(s![1.., ..;2]);
    assert_eq!(vi.shape(), &[2, 2]);

    // trailing comma
    let vi = a.slice(s![1.., ..;2, ]);
    assert_eq!(vi.shape(), &[2, 2]);
}
