
use alloc::vec::Vec;

use crate::imp_prelude::*;
use crate::dimension;
use crate::error::{ErrorKind, ShapeError};
use crate::OwnedRepr;
use crate::Zip;

/// Methods specific to `Array0`.
///
/// ***See also all methods for [`ArrayBase`]***
///
/// [`ArrayBase`]: struct.ArrayBase.html
impl<A> Array<A, Ix0> {
    /// Returns the single element in the array without cloning it.
    ///
    /// ```
    /// use ndarray::{arr0, Array0};
    ///
    /// // `Foo` doesn't implement `Clone`.
    /// #[derive(Debug, Eq, PartialEq)]
    /// struct Foo;
    ///
    /// let array: Array0<Foo> = arr0(Foo);
    /// let scalar: Foo = array.into_scalar();
    /// assert_eq!(scalar, Foo);
    /// ```
    pub fn into_scalar(self) -> A {
        let size = ::std::mem::size_of::<A>();
        if size == 0 {
            // Any index in the `Vec` is fine since all elements are identical.
            self.data.into_vec().remove(0)
        } else {
            // Find the index in the `Vec` corresponding to `self.ptr`.
            // (This is necessary because the element in the array might not be
            // the first element in the `Vec`, such as if the array was created
            // by `array![1, 2, 3, 4].slice_move(s![2])`.)
            let first = self.ptr.as_ptr() as usize;
            let base = self.data.as_ptr() as usize;
            let index = (first - base) / size;
            debug_assert_eq!((first - base) % size, 0);
            // Remove the element at the index and return it.
            self.data.into_vec().remove(index)
        }
    }
}

/// Methods specific to `Array`.
///
/// ***See also all methods for [`ArrayBase`]***
///
/// [`ArrayBase`]: struct.ArrayBase.html
impl<A, D> Array<A, D>
where
    D: Dimension,
{
    /// Return a vector of the elements in the array, in the way they are
    /// stored internally.
    ///
    /// If the array is in standard memory layout, the logical element order
    /// of the array (`.iter()` order) and of the returned vector will be the same.
    pub fn into_raw_vec(self) -> Vec<A> {
        self.data.into_vec()
    }
}

/// Methods specific to `Array2`.
///
/// ***See also all methods for [`ArrayBase`]***
///
/// [`ArrayBase`]: struct.ArrayBase.html
impl<A> Array<A, Ix2> {
    /// Append a row to an array with row major memory layout.
    ///
    /// ***Errors*** with a layout error if the array is not in standard order or
    /// if it has holes, even exterior holes (from slicing). <br>
    /// ***Errors*** with shape error if the length of the input row does not match
    /// the length of the rows in the array. <br>
    ///
    /// The memory layout matters, since it determines in which direction the array can easily
    /// grow. Notice that an empty array is compatible both ways. The amortized average
    /// complexity of the append is O(m) where *m* is the length of the row.
    ///
    /// ```rust
    /// use ndarray::{Array, ArrayView, array};
    ///
    /// // create an empty array and append
    /// let mut a = Array::zeros((0, 4));
    /// a.try_append_row(ArrayView::from(&[ 1.,  2.,  3.,  4.])).unwrap();
    /// a.try_append_row(ArrayView::from(&[-1., -2., -3., -4.])).unwrap();
    ///
    /// assert_eq!(
    ///     a,
    ///     array![[ 1.,  2.,  3.,  4.],
    ///            [-1., -2., -3., -4.]]);
    /// ```
    pub fn try_append_row(&mut self, row: ArrayView<A, Ix1>) -> Result<(), ShapeError>
    where
        A: Clone,
    {
        let row_len = row.len();
        if row_len != self.len_of(Axis(1)) {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
        let mut res_dim = self.raw_dim();
        res_dim[0] += 1;
        let new_len = dimension::size_of_shape_checked(&res_dim)?;

        // array must be c-contiguous and be "full" (have no exterior holes)
        if !self.is_standard_layout() || self.len() != self.data.len() {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleLayout));
        }

        unsafe {
            // grow backing storage and update head ptr
            debug_assert_eq!(self.data.as_ptr(), self.as_ptr());
            self.ptr = self.data.reserve(row_len);  // because we are standard order

            // recompute strides - if the array was previously empty, it could have
            // zeros in strides.
            let strides = res_dim.default_strides();

            // copy elements from view to the array now
            //
            // make a raw view with the new row
            // safe because the data was "full"
            let tail_ptr = self.data.as_end_nonnull();
            let tail_view = RawArrayViewMut::new(tail_ptr, Ix1(row_len), Ix1(1));

            struct SetLenOnDrop<'a, A: 'a> {
                len: usize,
                data: &'a mut OwnedRepr<A>,
            }

            let mut length_guard = SetLenOnDrop {
                len: self.data.len(),
                data: &mut self.data,
            };

            impl<A> Drop for SetLenOnDrop<'_, A> {
                fn drop(&mut self) {
                    unsafe {
                        self.data.set_len(self.len);
                    }
                }
            }

            // assign the new elements
            Zip::from(tail_view).and(row)
                .for_each(|to, from| {
                    to.write(from.clone());
                    length_guard.len += 1;
                });

            drop(length_guard);

            // update array dimension
            self.strides = strides;
            self.dim[0] += 1;

        }
        // multiple assertions after pointer & dimension update
        debug_assert_eq!(self.data.len(), self.len());
        debug_assert_eq!(self.len(), new_len);
        debug_assert!(self.is_standard_layout());

        Ok(())
    }

    /// Append a column to an array with column major memory layout.
    ///
    /// ***Errors*** with a layout error if the array is not in column major order or
    /// if it has holes, even exterior holes (from slicing). <br>
    /// ***Errors*** with shape error if the length of the input column does not match
    /// the length of the columns in the array.<br>
    ///
    /// The memory layout matters, since it determines in which direction the array can easily
    /// grow. Notice that an empty array is compatible both ways. The amortized average
    /// complexity of the append is O(m) where *m* is the length of the column.
    ///
    /// ```rust
    /// use ndarray::{Array, ArrayView, array};
    ///
    /// // create an empty array and append
    /// let mut a = Array::zeros((2, 0));
    /// a.try_append_column(ArrayView::from(&[1., 2.])).unwrap();
    /// a.try_append_column(ArrayView::from(&[-1., -2.])).unwrap();
    ///
    /// assert_eq!(
    ///     a,
    ///     array![[1., -1.],
    ///            [2., -2.]]);
    /// ```
    pub fn try_append_column(&mut self, column: ArrayView<A, Ix1>) -> Result<(), ShapeError>
    where
        A: Clone,
    {
        self.swap_axes(0, 1);
        let ret = self.try_append_row(column);
        self.swap_axes(0, 1);
        ret
    }
}

impl<A, D> Array<A, D>
    where D: Dimension
{
    /// Append an array to the array
    ///
    /// The axis-to-append-to `axis` must be the array's "growing axis" for this operation
    /// to succeed. The growing axis is the outermost or last-visited when elements are visited in
    /// memory order:
    ///
    /// `axis` must be the growing axis of the current array, an axis with length 0 or 1.
    ///
    /// - This is the 0th axis for standard layout arrays
    /// - This is the *n*-1 th axis for fortran layout arrays
    /// - If the array is empty (the axis or any other has length 0) or if `axis`
    ///   has length 1, then the array can always be appended.
    ///
    /// ***Errors*** with a layout error if the array is not in standard order or
    /// if it has holes, even exterior holes (from slicing). <br>
    /// ***Errors*** with shape error if the length of the input row does not match
    /// the length of the rows in the array. <br>
    ///
    /// The memory layout of the `self` array matters, since it determines in which direction the
    /// array can easily grow. Notice that an empty array is compatible both ways. The amortized
    /// average complexity of the append is O(m) where *m* is the number of elements in the
    /// array-to-append (equivalent to how `Vec::extend` works).
    ///
    /// The memory layout of the argument `array` does not matter.
    ///
    /// ```rust
    /// use ndarray::{Array, ArrayView, array, Axis};
    ///
    /// // create an empty array and append
    /// let mut a = Array::zeros((0, 4));
    /// let ones  = ArrayView::from(&[1.; 8]).into_shape((2, 4)).unwrap();
    /// let zeros = ArrayView::from(&[0.; 8]).into_shape((2, 4)).unwrap();
    /// a.try_append_array(Axis(0), ones).unwrap();
    /// a.try_append_array(Axis(0), zeros).unwrap();
    /// a.try_append_array(Axis(0), ones).unwrap();
    ///
    /// assert_eq!(
    ///     a,
    ///     array![[1., 1., 1., 1.],
    ///            [1., 1., 1., 1.],
    ///            [0., 0., 0., 0.],
    ///            [0., 0., 0., 0.],
    ///            [1., 1., 1., 1.],
    ///            [1., 1., 1., 1.]]);
    /// ```
    pub fn try_append_array(&mut self, axis: Axis, mut array: ArrayView<A, D>)
        -> Result<(), ShapeError>
    where
        A: Clone,
        D: RemoveAxis,
    {
        if self.ndim() == 0 {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }

        let current_axis_len = self.len_of(axis);
        let remaining_shape = self.raw_dim().remove_axis(axis);
        let array_rem_shape = array.raw_dim().remove_axis(axis);

        if remaining_shape != array_rem_shape {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }

        let len_to_append = array.len();
        if len_to_append == 0 {
            return Ok(());
        }

        let array_shape = array.raw_dim();
        let mut res_dim = self.raw_dim();
        res_dim[axis.index()] += array_shape[axis.index()];
        let new_len = dimension::size_of_shape_checked(&res_dim)?;

        let self_is_empty = self.is_empty();

        // array must be empty or have `axis` as the outermost (longest stride) axis
        if !self_is_empty && current_axis_len > 1 {
            // `axis` must be max stride axis or equal to its stride
            let max_stride_axis = self.axes().max_by_key(|ax| ax.stride).unwrap();
            if max_stride_axis.axis != axis && max_stride_axis.stride > self.stride_of(axis) {
                return Err(ShapeError::from_kind(ErrorKind::IncompatibleLayout));
            }
        }

        // array must be be "full" (have no exterior holes)
        if self.len() != self.data.len() {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleLayout));
        }

        let strides = if self_is_empty {
            // recompute strides - if the array was previously empty, it could have zeros in
            // strides.
            // The new order is based on c/f-contig but must have `axis` as outermost axis.
            if axis == Axis(self.ndim() - 1) {
                // prefer f-contig when appending to the last axis
                // Axis n - 1 is outermost axis
                res_dim.fortran_strides()
            } else {
                // Default with modification
                res_dim.slice_mut().swap(0, axis.index());
                let mut strides = res_dim.default_strides();
                res_dim.slice_mut().swap(0, axis.index());
                strides.slice_mut().swap(0, axis.index());
                strides
            }
        } else if current_axis_len == 1 {
            // This is the outermost/longest stride axis; so we find the max across the other axes
            let new_stride = self.axes().fold(1, |acc, ax| {
                if ax.axis == axis { acc } else {
                    Ord::max(acc, ax.len as isize * ax.stride)
                }
            });
            let mut strides = self.strides.clone();
            strides[axis.index()] = new_stride as usize;
            strides
        } else {
            self.strides.clone()
        };

        unsafe {
            // grow backing storage and update head ptr
            debug_assert_eq!(self.data.as_ptr(), self.as_ptr());
            self.ptr = self.data.reserve(len_to_append); // because we are standard order

            // copy elements from view to the array now
            //
            // make a raw view with the new row
            // safe because the data was "full"
            let tail_ptr = self.data.as_end_nonnull();
            let mut tail_view = RawArrayViewMut::new(tail_ptr, array_shape, strides.clone());

            struct SetLenOnDrop<'a, A: 'a> {
                len: usize,
                data: &'a mut OwnedRepr<A>,
            }

            let mut length_guard = SetLenOnDrop {
                len: self.data.len(),
                data: &mut self.data,
            };

            impl<A> Drop for SetLenOnDrop<'_, A> {
                fn drop(&mut self) {
                    unsafe {
                        self.data.set_len(self.len);
                    }
                }
            }

            // To be robust for panics and drop the right elements, we want
            // to fill the tail in-order, so that we can drop the right elements on
            // panic.
            //
            // We have: Zip::from(tail_view).and(array)
            // Transform tail_view into standard order by inverting and moving its axes.
            // Keep the Zip traversal unchanged by applying the same axis transformations to
            // `array`. This ensures the Zip traverses the underlying memory in order.
            //
            // XXX It would be possible to skip this transformation if the element
            // doesn't have drop. However, in the interest of code coverage, all elements
            // use this code initially.

            if tail_view.ndim() > 1 {
                for i in 0..tail_view.ndim() {
                    if tail_view.stride_of(Axis(i)) < 0 {
                        tail_view.invert_axis(Axis(i));
                        array.invert_axis(Axis(i));
                    }
                }
                sort_axes_to_standard_order(&mut tail_view, &mut array);
            } 
            Zip::from(tail_view).and(array)
                .debug_assert_c_order()
                .for_each(|to, from| {
                    to.write(from.clone());
                    length_guard.len += 1;
                });

            drop(length_guard);

            // update array dimension
            self.strides = strides;
            self.dim = res_dim;
        }
        // multiple assertions after pointer & dimension update
        debug_assert_eq!(self.data.len(), self.len());
        debug_assert_eq!(self.len(), new_len);

        Ok(())
    }
}

fn sort_axes_to_standard_order<S, S2, D>(a: &mut ArrayBase<S, D>, b: &mut ArrayBase<S2, D>)
where
    S: RawData,
    S2: RawData,
    D: Dimension,
{
    if a.ndim() <= 1 {
        return;
    }
    sort_axes_impl(&mut a.dim, &mut a.strides, &mut b.dim, &mut b.strides);
    debug_assert!(a.is_standard_layout(), "not std layout dim: {:?}, strides: {:?}",
                  a.shape(), a.strides());
}

fn sort_axes_impl<D>(adim: &mut D, astrides: &mut D, bdim: &mut D, bstrides: &mut D)
where
    D: Dimension,
{
    debug_assert!(adim.ndim() > 1);
    debug_assert_eq!(adim.ndim(), bdim.ndim());
    // bubble sort axes
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..adim.ndim() - 1 {
            let axis_i = i;
            let next_axis = i + 1;

            // make sure higher stride axes sort before.
            debug_assert!(astrides.slice()[axis_i] as isize >= 0);
            if (astrides.slice()[axis_i] as isize) < astrides.slice()[next_axis] as isize {
                changed = true;
                adim.slice_mut().swap(axis_i, next_axis);
                astrides.slice_mut().swap(axis_i, next_axis);
                bdim.slice_mut().swap(axis_i, next_axis);
                bstrides.slice_mut().swap(axis_i, next_axis);
            }
        }
    }
}

