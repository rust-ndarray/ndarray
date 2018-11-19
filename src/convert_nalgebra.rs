use dimension;
use imp_prelude::*;
use nalgebra as na;
use std::isize;

/// **Requires crate feature `"nalgebra-0_16"`**
impl<A, R, S1, S2> From<na::Matrix<A, R, na::U1, S1>> for ArrayBase<S2, Ix1>
where
    A: na::Scalar,
    R: na::Dim,
    S1: na::storage::Storage<A, R, na::U1>,
    S2: DataOwned<Elem = A>,
{
    /// Converts the `nalgebra::Vector` to `ndarray::ArrayBase`.
    ///
    /// **Panics** if the number of elements overflows `isize`.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate ndarray;
    ///
    /// use nalgebra::Vector3;
    /// use ndarray::{array, Array1};
    ///
    /// # fn main() {
    /// let vector = Vector3::new(1, 2, 3);
    /// let array = Array1::from(vector);
    /// assert_eq!(array, array![1, 2, 3]);
    /// # }
    /// ```
    fn from(vector: na::Matrix<A, R, na::U1, S1>) -> ArrayBase<S2, Ix1> {
        ArrayBase::from_vec(vector.iter().cloned().collect())
    }
}

/// **Requires crate feature `"nalgebra-0_16"`**
impl<'a, A, R, RStride, CStride> From<na::MatrixSlice<'a, A, R, na::U1, RStride, CStride>>
    for ArrayView<'a, A, Ix1>
where
    A: na::Scalar,
    R: na::Dim,
    RStride: na::Dim,
    CStride: na::Dim,
{
    /// Converts the 1-D `nalgebra::MatrixSlice` to `ndarray::ArrayView`.
    ///
    /// **Panics** if the number of elements, row stride, or size in bytes
    /// overflows `isize`.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate ndarray;
    ///
    /// use nalgebra::MatrixSlice3x1;
    /// use ndarray::{array, ArrayView1};
    ///
    /// # fn main() {
    /// let slice = MatrixSlice3x1::from_slice(&[1, 2, 3]);
    /// let view = ArrayView1::from(slice);
    /// assert_eq!(view, array![1, 2, 3]);
    /// # }
    /// ```
    fn from(slice: na::MatrixSlice<'a, A, R, na::U1, RStride, CStride>) -> ArrayView<'a, A, Ix1> {
        if slice.is_empty() {
            ArrayView::from_shape(slice.shape().0, &[]).unwrap()
        } else {
            let dim = Dim(slice.shape().0);
            let strides = Dim(slice.strides().0);
            ndassert!(
                strides[0] <= isize::MAX as usize,
                "stride {} must not exceed `isize::MAX`",
                strides[0],
            );
            dimension::max_abs_offset_check_overflow::<A, _>(&dim, &strides)
                .expect("overflow converting `nalgebra::MatrixSlice` to `nalgebra::ArrayView`");
            let ptr: *const A = slice.iter().next().unwrap();
            unsafe { ArrayView::from_shape_ptr(dim.strides(strides), ptr) }
        }
    }
}

/// **Requires crate feature `"nalgebra-0_16"`**
impl<'a, A, R, RStride, CStride> From<na::MatrixSliceMut<'a, A, R, na::U1, RStride, CStride>>
    for ArrayViewMut<'a, A, Ix1>
where
    A: na::Scalar,
    R: na::Dim,
    RStride: na::Dim,
    CStride: na::Dim,
{
    /// Converts the 1-D `nalgebra::MatrixSliceMut` to `ndarray::ArrayViewMut`.
    ///
    /// **Panics** if the number of elements, row stride, or size in bytes
    /// overflows `isize`. Also panics if the row stride is zero when there is
    /// more than one row.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate ndarray;
    ///
    /// use nalgebra::MatrixSliceMut3x1;
    /// use ndarray::{array, ArrayViewMut1};
    ///
    /// # fn main() {
    /// // `from_slice` assumes column-major memory layout.
    /// let mut data = [1, 2, 3];
    /// let slice = MatrixSliceMut3x1::from_slice(&mut data);
    /// let view = ArrayViewMut1::from(slice);
    /// assert_eq!(view, array![1, 2, 3]);
    /// # }
    /// ```
    fn from(
        mut slice: na::MatrixSliceMut<'a, A, R, na::U1, RStride, CStride>,
    ) -> ArrayViewMut<'a, A, Ix1> {
        if slice.is_empty() {
            ArrayViewMut::from_shape(slice.shape().0, &mut []).unwrap()
        } else {
            let dim = Dim(slice.shape().0);
            let strides = Dim(slice.strides().0);
            ndassert!(
                strides[0] <= isize::MAX as usize,
                "stride {} must not exceed `isize::MAX`",
                strides[0],
            );
            // `nalgebra` should prevent this ever being violated but currently
            // doesn't (rustsim/nalgebra#473).
            ndassert!(
                dim[0] <= 1 || strides[0] != 0,
                "stride {} must be nonzero when axis length {} is > 1",
                strides[0],
                dim[0],
            );
            dimension::max_abs_offset_check_overflow::<A, _>(&dim, &strides).expect(
                "overflow converting `nalgebra::MatrixSliceMut` to `nalgebra::ArrayViewMut`",
            );
            let ptr: *mut A = slice.iter_mut().next().unwrap();
            unsafe { ArrayViewMut::from_shape_ptr(dim.strides(strides), ptr) }
        }
    }
}

/// **Requires crate feature `"nalgebra-0_16"`**
impl<A, R, C, S1, S2> From<na::Matrix<A, R, C, S1>> for ArrayBase<S2, Ix2>
where
    A: na::Scalar,
    R: na::Dim,
    C: na::Dim,
    S1: na::storage::Storage<A, R, C>,
    S2: DataOwned<Elem = A>,
{
    /// Converts the `nalgebra::Matrix` to `ndarray::ArrayBase`.
    ///
    /// **Panics** if the number of rows, columns, or elements overflows `isize`.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate ndarray;
    ///
    /// use nalgebra::Matrix2x3;
    /// use ndarray::{array, Array2};
    ///
    /// # fn main() {
    /// let matrix = Matrix2x3::new(1, 2, 3, 4, 5, 6);
    /// let array = Array2::from(matrix);
    /// assert_eq!(array, array![[1, 2, 3], [4, 5, 6]]);
    /// # }
    /// ```
    fn from(matrix: na::Matrix<A, R, C, S1>) -> ArrayBase<S2, Ix2> {
        let (rows, cols) = matrix.shape();
        ArrayBase::from_shape_vec((cols, rows), matrix.iter().cloned().collect())
            .expect("convert `nalgebra::Matrix` to `ndarray::ArrayBase`")
            .reversed_axes()
    }
}

/// **Requires crate feature `"nalgebra-0_16"`**
impl<'a, A, R, C, RStride, CStride> From<na::MatrixSlice<'a, A, R, C, RStride, CStride>>
    for ArrayView<'a, A, Ix2>
where
    A: na::Scalar,
    R: na::Dim,
    C: na::Dim,
    RStride: na::Dim,
    CStride: na::Dim,
{
    /// Converts the `nalgebra::MatrixSlice` to `ndarray::ArrayView`.
    ///
    /// **Panics** if the number of rows, number of columns, row stride, column
    /// stride, number of elements, or size in bytes overflows `isize`.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate ndarray;
    ///
    /// use nalgebra::MatrixSlice2x3;
    /// use ndarray::{array, ArrayView2};
    ///
    /// # fn main() {
    /// // `from_slice` assumes column-major memory layout.
    /// let slice = MatrixSlice2x3::from_slice(&[1, 4, 2, 5, 3, 6]);
    /// let view = ArrayView2::from(slice);
    /// assert_eq!(view, array![[1, 2, 3], [4, 5, 6]]);
    /// # }
    /// ```
    fn from(slice: na::MatrixSlice<'a, A, R, C, RStride, CStride>) -> ArrayView<'a, A, Ix2> {
        if slice.is_empty() {
            ArrayView::from_shape(slice.shape(), &[]).unwrap()
        } else {
            let dim = Dim(slice.shape());
            let strides = Dim(slice.strides());
            ndassert!(
                strides[0] <= isize::MAX as usize && strides[1] <= isize::MAX as usize,
                "strides {:?} must not exceed `isize::MAX`",
                strides,
            );
            dimension::max_abs_offset_check_overflow::<A, _>(&dim, &strides)
                .expect("overflow converting `nalgebra::MatrixSlice` to `nalgebra::ArrayView`");
            let ptr: *const A = slice.iter().next().unwrap();
            unsafe { ArrayView::from_shape_ptr(dim.strides(strides), ptr) }
        }
    }
}

/// **Requires crate feature `"nalgebra-0_16"`**
impl<'a, A, R, C, RStride, CStride> From<na::MatrixSliceMut<'a, A, R, C, RStride, CStride>>
    for ArrayViewMut<'a, A, Ix2>
where
    A: na::Scalar,
    R: na::Dim,
    C: na::Dim,
    RStride: na::Dim,
    CStride: na::Dim,
{
    /// Converts the `nalgebra::MatrixSliceMut` to `ndarray::ArrayViewMut`.
    ///
    /// **Panics** if the number of rows, number of columns, row stride, column
    /// stride, number of elements, or size in bytes overflows `isize`. Also
    /// panics if the row stride or column stride is zero when the length of
    /// the corresponding axis is greater than one.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate nalgebra;
    /// # extern crate ndarray;
    ///
    /// use nalgebra::MatrixSliceMut2x3;
    /// use ndarray::{array, ArrayViewMut2};
    ///
    /// # fn main() {
    /// // `from_slice` assumes column-major memory layout.
    /// let mut data = [1, 4, 2, 5, 3, 6];
    /// let slice = MatrixSliceMut2x3::from_slice(&mut data);
    /// let view = ArrayViewMut2::from(slice);
    /// assert_eq!(view, array![[1, 2, 3], [4, 5, 6]]);
    /// # }
    /// ```
    fn from(
        mut slice: na::MatrixSliceMut<'a, A, R, C, RStride, CStride>,
    ) -> ArrayViewMut<'a, A, Ix2> {
        if slice.is_empty() {
            ArrayViewMut::from_shape(slice.shape(), &mut []).unwrap()
        } else {
            let dim = Dim(slice.shape());
            let strides = Dim(slice.strides());
            ndassert!(
                strides[0] <= isize::MAX as usize && strides[1] <= isize::MAX as usize,
                "strides {:?} must not exceed `isize::MAX`",
                strides,
            );
            // `nalgebra` should prevent this ever being violated but currently
            // doesn't (rustsim/nalgebra#473).
            ndassert!(
                (dim[0] <= 1 || strides[0] != 0) && (dim[1] <= 1 || strides[1] != 0),
                "strides {:?} must be nonzero when corresponding lengths {:?} are > 1",
                strides,
                dim,
            );
            dimension::max_abs_offset_check_overflow::<A, _>(&dim, &strides).expect(
                "overflow converting `nalgebra::MatrixSliceMut` to `nalgebra::ArrayViewMut`",
            );
            let ptr: *mut A = slice.iter_mut().next().unwrap();
            unsafe { ArrayViewMut::from_shape_ptr(dim.strides(strides), ptr) }
        }
    }
}
