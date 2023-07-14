use dlpark::prelude::*;
use ndarray::ManagedArray;

#[test]
fn test_dlpack() {
    let arr = ndarray::arr1(&[1i32, 2, 3]);
    let ptr = arr.as_ptr();
    let dlpack = arr.to_dlpack();
    let arr2 = ManagedArray::<i32, _>::from_dlpack(dlpack);
    let ptr2 = arr2.as_ptr();
    assert_eq!(ptr, ptr2);
    // dbg!(&arr2);
    let arr3 = arr2.to_owned();
    // dbg!(&arr3);
    let ptr3 = arr3.as_ptr();
    assert_ne!(ptr2, ptr3);
}
