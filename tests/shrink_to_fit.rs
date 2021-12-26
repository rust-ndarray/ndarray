use ndarray::{s, Array};

#[test]
fn dim_0() {
    let mut raw_vec = Vec::new();
    for i in 0..4 * 5 * 6 {
        raw_vec.push(i);
    }
    let a = Array::from_shape_vec((4, 5, 6), raw_vec).unwrap();
    let mut a_slice = a.slice_move(s![0..2, .., ..]);
    let a_slice_clone = a_slice.view().to_owned();
    a_slice.shrink_to_fit();
    assert_eq!(a_slice, a_slice_clone);
}

#[test]
fn swap_axis_dim_0() {
    let mut raw_vec = Vec::new();
    for i in 0..4 * 5 * 6 {
        raw_vec.push(i);
    }
    let mut a = Array::from_shape_vec((4, 5, 6), raw_vec).unwrap();
    a.swap_axes(0, 1);
    let mut a_slice = a.slice_move(s![2..3, .., ..]);
    let a_slice_clone = a_slice.view().to_owned();
    a_slice.shrink_to_fit();
    assert_eq!(a_slice, a_slice_clone);
}

#[test]
fn swap_axis_dim() {
    let mut raw_vec = Vec::new();
    for i in 0..4 * 5 * 6 {
        raw_vec.push(i);
    }
    let mut a = Array::from_shape_vec((4, 5, 6), raw_vec).unwrap();
    a.swap_axes(2, 1);
    let mut a_slice = a.slice_move(s![2..3, 0..3, 0..;2]);
    let a_slice_clone = a_slice.view().to_owned();
    a_slice.shrink_to_fit();
    assert_eq!(a_slice, a_slice_clone);
}

#[test]
fn stride_negative() {
    let mut raw_vec = Vec::new();
    for i in 0..4 * 5 * 6 {
        raw_vec.push(i);
    }
    let a = Array::from_shape_vec((4, 5, 6), raw_vec).unwrap();
    let mut a_slice = a.slice_move(s![2..3, 0..3, 0..;-1]);
    let a_slice_clone = a_slice.view().to_owned();
    a_slice.shrink_to_fit();
    assert_eq!(a_slice, a_slice_clone);
}
