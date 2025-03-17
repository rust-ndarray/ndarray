extern crate blas_src;
use ndarray::{Array1, Array2, ArrayD, linalg::Dot, Ix1, Ix2};

#[test]
fn test_arrayd_dot_2d() {
    let mat1 = ArrayD::from_shape_vec(vec![3, 2], vec![3.0; 6]).unwrap();
    let mat2 = ArrayD::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();

    let result = mat1.dot(&mat2);

    // Verify the result is correct
    assert_eq!(result.ndim(), 2);
    assert_eq!(result.shape(), &[3, 3]);

    // Compare with Array2 implementation
    let mat1_2d = Array2::from_shape_vec((3, 2), vec![3.0; 6]).unwrap();
    let mat2_2d = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
    let expected = mat1_2d.dot(&mat2_2d);

    assert_eq!(result.into_dimensionality::<Ix2>().unwrap(), expected);
}

#[test]
fn test_arrayd_dot_1d() {
    // Test 1D array dot product
    let vec1 = ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let vec2 = ArrayD::from_shape_vec(vec![3], vec![4.0, 5.0, 6.0]).unwrap();

    let result = vec1.dot(&vec2);

    // Verify scalar result
    assert_eq!(result.ndim(), 0);
    assert_eq!(result.shape(), &[]);
    assert_eq!(result[[]], 32.0); // 1*4 + 2*5 + 3*6
}

#[test]
#[should_panic(expected = "Dot product for ArrayD is only supported for 1D and 2D arrays")]
fn test_arrayd_dot_3d() {
    // Test that 3D arrays are not supported
    let arr1 = ArrayD::from_shape_vec(vec![2, 2, 2], vec![1.0; 8]).unwrap();
    let arr2 = ArrayD::from_shape_vec(vec![2, 2, 2], vec![1.0; 8]).unwrap();

    let _result = arr1.dot(&arr2); // Should panic
}

#[test]
#[should_panic(expected = "ndarray: inputs 2 × 3 and 4 × 5 are not compatible for matrix multiplication")]
fn test_arrayd_dot_incompatible_dims() {
    // Test arrays with incompatible dimensions
    let arr1 = ArrayD::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
    let arr2 = ArrayD::from_shape_vec(vec![4, 5], vec![1.0; 20]).unwrap();

    let _result = arr1.dot(&arr2); // Should panic
}

#[test]
fn test_arrayd_dot_matrix_vector() {
    // Test matrix-vector multiplication
    let mat = ArrayD::from_shape_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let vec = ArrayD::from_shape_vec(vec![2], vec![1.0, 2.0]).unwrap();

    let result = mat.dot(&vec);

    // Verify result
    assert_eq!(result.ndim(), 1);
    assert_eq!(result.shape(), &[3]);

    // Compare with Array2 implementation
    let mat_2d = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let vec_1d = Array1::from_vec(vec![1.0, 2.0]);
    let expected = mat_2d.dot(&vec_1d);

    assert_eq!(result.into_dimensionality::<Ix1>().unwrap(), expected);
} 