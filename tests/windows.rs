#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names
)]

use ndarray::prelude::*;
use ndarray::{arr3, Zip};

// Edge Cases for Windows iterator:
//
// - window size is 0
//     - what is the behaviour of the standard for this situation?
//       "Panics if size is 0."
// - window size of 1
//     - should a warning be printed?
//     - what is the behaviour of the standard for this situation?
//       => No overlapping for size-1-windows but normal behaviour besides that.
// - window size bigger than actual array size
//     - ragged windows or panic?
//     - what is the behaviour of the standard for this situation?
//       "If the slice is shorter than size, the iterator returns no values."

/// Test that verifies the `Windows` iterator panics on window sizes equal to zero.
#[test]
#[should_panic]
fn windows_iterator_zero_size() {
    let a = Array::from_iter(10..37).into_shape((3, 3, 3)).unwrap();
    a.windows(Dim((0, 0, 0)));
}

/// Test that verifites that no windows are yielded on oversized window sizes.
#[test]
fn windows_iterator_oversized() {
    let a = Array::from_iter(10..37).into_shape((3, 3, 3)).unwrap();
    let mut iter = a.windows((4, 3, 2)).into_iter(); // (4,3,2) doesn't fit into (3,3,3) => oversized!
    assert_eq!(iter.next(), None);
}

/// Simple test for iterating 1d-arrays via `Windows`.
#[test]
fn windows_iterator_1d() {
    let a = Array::from_iter(10..20).into_shape(10).unwrap();
    itertools::assert_equal(
        a.windows(Dim(4)),
        vec![
            arr1(&[10, 11, 12, 13]),
            arr1(&[11, 12, 13, 14]),
            arr1(&[12, 13, 14, 15]),
            arr1(&[13, 14, 15, 16]),
            arr1(&[14, 15, 16, 17]),
            arr1(&[15, 16, 17, 18]),
            arr1(&[16, 17, 18, 19]),
        ],
    );
}

/// Simple test for iterating 2d-arrays via `Windows`.
#[test]
fn windows_iterator_2d() {
    let a = Array::from_iter(10..30).into_shape((5, 4)).unwrap();
    itertools::assert_equal(
        a.windows(Dim((3, 2))),
        vec![
            arr2(&[[10, 11], [14, 15], [18, 19]]),
            arr2(&[[11, 12], [15, 16], [19, 20]]),
            arr2(&[[12, 13], [16, 17], [20, 21]]),
            arr2(&[[14, 15], [18, 19], [22, 23]]),
            arr2(&[[15, 16], [19, 20], [23, 24]]),
            arr2(&[[16, 17], [20, 21], [24, 25]]),
            arr2(&[[18, 19], [22, 23], [26, 27]]),
            arr2(&[[19, 20], [23, 24], [27, 28]]),
            arr2(&[[20, 21], [24, 25], [28, 29]]),
        ],
    );
}

/// Simple test for iterating 3d-arrays via `Windows`.
#[test]
fn windows_iterator_3d() {
    let a = Array::from_iter(10..37).into_shape((3, 3, 3)).unwrap();
    itertools::assert_equal(
        a.windows(Dim((2, 2, 2))),
        vec![
            arr3(&[[[10, 11], [13, 14]], [[19, 20], [22, 23]]]),
            arr3(&[[[11, 12], [14, 15]], [[20, 21], [23, 24]]]),
            arr3(&[[[13, 14], [16, 17]], [[22, 23], [25, 26]]]),
            arr3(&[[[14, 15], [17, 18]], [[23, 24], [26, 27]]]),
            arr3(&[[[19, 20], [22, 23]], [[28, 29], [31, 32]]]),
            arr3(&[[[20, 21], [23, 24]], [[29, 30], [32, 33]]]),
            arr3(&[[[22, 23], [25, 26]], [[31, 32], [34, 35]]]),
            arr3(&[[[23, 24], [26, 27]], [[32, 33], [35, 36]]]),
        ],
    );
}

#[test]
fn test_window_zip() {
    let a = Array::from_iter(0..64).into_shape((4, 4, 4)).unwrap();

    for x in 1..4 {
        for y in 1..4 {
            for z in 1..4 {
                Zip::indexed(a.windows((x, y, z))).for_each(|(i, j, k), window| {
                    let x = x as isize;
                    let y = y as isize;
                    let z = z as isize;
                    let i = i as isize;
                    let j = j as isize;
                    let k = k as isize;
                    assert_eq!(window, a.slice(s![i..i + x, j..j + y, k..k + z]));
                })
            }
        }
    }
}

/// Test verifies that non existent Axis results in panic
#[test]
#[should_panic]
fn axis_windows_outofbound() {
    let a = Array::from_iter(10..37).into_shape((3, 3, 3)).unwrap();
    a.axis_windows(Axis(4), 2);
}

/// Test verifies that zero sizes results in panic
#[test]
#[should_panic]
fn axis_windows_zero_size() {
    let a = Array::from_iter(10..37).into_shape((3, 3, 3)).unwrap();
    a.axis_windows(Axis(0), 0);
}

/// Test verifies that over sized windows yield nothing
#[test]
fn axis_windows_oversized() {
    let a = Array::from_iter(10..37).into_shape((3, 3, 3)).unwrap();
    let mut iter = a.axis_windows(Axis(2), 4).into_iter();
    assert_eq!(iter.next(), None);
}

/// Simple test for iterating 1d-arrays via `Axis Windows`.
#[test]
fn test_axis_windows_1d() {
    let a = Array::from_iter(10..20).into_shape(10).unwrap();

    itertools::assert_equal(
        a.axis_windows(Axis(0), 5),
        vec![
            arr1(&[10, 11, 12, 13, 14]),
            arr1(&[11, 12, 13, 14, 15]),
            arr1(&[12, 13, 14, 15, 16]),
            arr1(&[13, 14, 15, 16, 17]),
            arr1(&[14, 15, 16, 17, 18]),
            arr1(&[15, 16, 17, 18, 19]),
        ],
    );
}

/// Simple test for iterating 2d-arrays via `Axis Windows`.
#[test]
fn test_axis_windows_2d() {
    let a = Array::from_iter(10..30).into_shape((5, 4)).unwrap();

    itertools::assert_equal(
        a.axis_windows(Axis(0), 2),
        vec![
            arr2(&[[10, 11, 12, 13], [14, 15, 16, 17]]),
            arr2(&[[14, 15, 16, 17], [18, 19, 20, 21]]),
            arr2(&[[18, 19, 20, 21], [22, 23, 24, 25]]),
            arr2(&[[22, 23, 24, 25], [26, 27, 28, 29]]),
        ],
    );
}

/// Simple test for iterating 3d-arrays via `Axis Windows`.
#[test]
fn test_axis_windows_3d() {
    let a = Array::from_iter(0..27).into_shape((3, 3, 3)).unwrap();

    itertools::assert_equal(
        a.axis_windows(Axis(1), 2),
        vec![
            arr3(&[
                [[0, 1, 2], [3, 4, 5]],
                [[9, 10, 11], [12, 13, 14]],
                [[18, 19, 20], [21, 22, 23]],
            ]),
            arr3(&[
                [[3, 4, 5], [6, 7, 8]],
                [[12, 13, 14], [15, 16, 17]],
                [[21, 22, 23], [24, 25, 26]],
            ]),
        ],
    );
}


#[test]
fn test_window_neg_stride() {
    let array = Array::from_iter(1..10).into_shape((3, 3)).unwrap();

    // window neg/pos stride combinations
    
    // Make a 2 x 2 array of the windows of the 3 x 3 array
    // and compute test answers from here
    let mut answer = Array::from_iter(array.windows((2, 2)).into_iter().map(|a| a.to_owned()))
        .into_shape((2, 2)).unwrap();

    answer.invert_axis(Axis(1));
    answer.map_inplace(|a| a.invert_axis(Axis(1)));

    itertools::assert_equal(
        array.slice(s![.., ..;-1]).windows((2, 2)),
        answer.iter()
    );

    answer.invert_axis(Axis(0));
    answer.map_inplace(|a| a.invert_axis(Axis(0)));

    itertools::assert_equal(
        array.slice(s![..;-1, ..;-1]).windows((2, 2)),
        answer.iter()
    );

    answer.invert_axis(Axis(1));
    answer.map_inplace(|a| a.invert_axis(Axis(1)));

    itertools::assert_equal(
        array.slice(s![..;-1, ..]).windows((2, 2)),
        answer.iter()
    );
}
