// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp;
use std::ptr;
use std::slice;

use imp_prelude::*;

use arraytraits;
use dimension;
use iterators;
use error::{self, ShapeError};
use super::zipsl;
use {
    NdIndex,
    AxisChunksIter,
    AxisChunksIterMut,
    Elements,
    ElementsMut,
    Indexed,
    IndexedMut,
    InnerIter,
    InnerIterMut,
    OuterIter,
    OuterIterMut,
};

impl<A, S, D> ArrayBase<S, D> where S: Data<Elem=A>, D: Dimension
{
    /// Return the total number of elements in the array.
    pub fn len(&self) -> usize {
        self.dim.size()
    }

    /// Return the shape of the array.
    pub fn dim(&self) -> D {
        self.dim.clone()
    }

    /// Return the shape of the array as a slice.
    pub fn shape(&self) -> &[Ix] {
        self.dim.slice()
    }

    /// Return the strides of the array
    pub fn strides(&self) -> &[Ixs] {
        let s = self.strides.slice();
        // reinterpret unsigned integer as signed
        unsafe {
            slice::from_raw_parts(s.as_ptr() as *const _, s.len())
        }
    }

    /// Return the number of dimensions (axes) in the array
    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }

    /// Return a read-only view of the array
    pub fn view(&self) -> ArrayView<A, D> {
        debug_assert!(self.pointer_is_inbounds());
        unsafe {
            ArrayView::new_(self.ptr, self.dim.clone(), self.strides.clone())
        }
    }

    /// Return a read-write view of the array
    pub fn view_mut(&mut self) -> ArrayViewMut<A, D>
        where S: DataMut,
    {
        self.ensure_unique();
        unsafe {
            ArrayViewMut::new_(self.ptr, self.dim.clone(), self.strides.clone())
        }
    }

    /// Return an uniquely owned copy of the array
    pub fn to_owned(&self) -> OwnedArray<A, D>
        where A: Clone
    {
        let data = if let Some(slc) = self.as_slice() {
            slc.to_vec()
        } else {
            self.iter().cloned().collect()
        };
        unsafe {
            ArrayBase::from_vec_dim_unchecked(self.dim.clone(), data)
        }
    }

    /// Return a shared ownership (copy on write) array.
    pub fn to_shared(&self) -> RcArray<A, D>
        where A: Clone
    {
        // FIXME: Avoid copying if it’s already an RcArray.
        self.to_owned().into_shared()
    }

    /// Turn the array into a shared ownership (copy on write) array,
    /// without any copying.
    pub fn into_shared(self) -> RcArray<A, D>
        where S: DataOwned,
    {
        let data = self.data.into_shared();
        ArrayBase {
            data: data,
            ptr: self.ptr,
            dim: self.dim,
            strides: self.strides,
        }
    }

    /// Return an iterator of references to the elements of the array.
    ///
    /// Iterator element type is `&A`.
    pub fn iter(&self) -> Elements<A, D> {
        debug_assert!(self.pointer_is_inbounds());
        self.view().into_iter_()
    }

    /// Return an iterator of mutable references to the elements of the array.
    ///
    /// Iterator element type is `&mut A`.
    pub fn iter_mut(&mut self) -> ElementsMut<A, D>
        where S: DataMut,
    {
        self.ensure_unique();
        self.view_mut().into_iter_()
    }

    /// Return an iterator of indexes and references to the elements of the array.
    ///
    /// Iterator element type is `(D, &A)`.
    pub fn indexed_iter(&self) -> Indexed<A, D> {
        Indexed(self.view().into_elements_base())
    }

    /// Return an iterator of indexes and mutable references to the elements of the array.
    ///
    /// Iterator element type is `(D, &mut A)`.
    pub fn indexed_iter_mut(&mut self) -> IndexedMut<A, D>
        where S: DataMut,
    {
        IndexedMut(self.view_mut().into_elements_base())
    }


    /// Return a sliced array.
    ///
    /// See [*Slicing*](#slicing) for full documentation.
    /// See also [`D::SliceArg`].
    ///
    /// [`D::SliceArg`]: trait.Dimension.html#associatedtype.SliceArg
    ///
    /// **Panics** if an index is out of bounds or stride is zero.<br>
    /// (**Panics** if `D` is `Vec` and `indexes` does not match the number of array axes.)
    pub fn slice(&self, indexes: &D::SliceArg) -> ArrayView<A, D> {
        let mut arr = self.view();
        arr.islice(indexes);
        arr
    }

    /// Return a sliced read-write view of the array.
    ///
    /// See also [`D::SliceArg`].
    ///
    /// [`D::SliceArg`]: trait.Dimension.html#associatedtype.SliceArg
    ///
    /// **Panics** if an index is out of bounds or stride is zero.<br>
    /// (**Panics** if `D` is `Vec` and `indexes` does not match the number of array axes.)
    pub fn slice_mut(&mut self, indexes: &D::SliceArg) -> ArrayViewMut<A, D>
        where S: DataMut
    {
        let mut arr = self.view_mut();
        arr.islice(indexes);
        arr
    }

    /// Slice the array’s view in place.
    ///
    /// See also [`D::SliceArg`].
    ///
    /// [`D::SliceArg`]: trait.Dimension.html#associatedtype.SliceArg
    ///
    /// **Panics** if an index is out of bounds or stride is zero.<br>
    /// (**Panics** if `D` is `Vec` and `indexes` does not match the number of array axes.)
    pub fn islice(&mut self, indexes: &D::SliceArg) {
        let offset = Dimension::do_slices(&mut self.dim, &mut self.strides, indexes);
        unsafe {
            self.ptr = self.ptr.offset(offset);
        }
        debug_assert!(self.pointer_is_inbounds());
    }

    /// Return a reference to the element at `index`, or return `None`
    /// if the index is out of bounds.
    ///
    /// Arrays also support indexing syntax: `array[index]`.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    ///
    /// assert!(
    ///     a.get((0, 1)) == Some(&2.) &&
    ///     a.get((0, 2)) == None &&
    ///     a[(0, 1)] == 2. &&
    ///     a[[0, 1]] == 2.
    /// );
    /// ```
    pub fn get<I>(&self, index: I) -> Option<&A>
        where I: NdIndex<Dim=D>,
    {
        let ptr = self.ptr;
        index.index_checked(&self.dim, &self.strides)
             .map(move |offset| unsafe { &*ptr.offset(offset) })
    }

    /// Return a mutable reference to the element at `index`, or return `None`
    /// if the index is out of bounds.
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut A>
        where S: DataMut,
              I: NdIndex<Dim=D>,
    {
        self.ensure_unique();
        let ptr = self.ptr;
        index.index_checked(&self.dim, &self.strides)
             .map(move |offset| unsafe { &mut *ptr.offset(offset) })
    }

    /// Perform *unchecked* array indexing.
    ///
    /// Return a reference to the element at `index`.
    ///
    /// **Note:** only unchecked for non-debug builds of ndarray.
    #[inline]
    pub unsafe fn uget(&self, index: D) -> &A {
        arraytraits::debug_bounds_check(self, &index);
        let off = Dimension::stride_offset(&index, &self.strides);
        &*self.ptr.offset(off)
    }

    /// Perform *unchecked* array indexing.
    ///
    /// Return a mutable reference to the element at `index`.
    ///
    /// **Note:** Only unchecked for non-debug builds of ndarray.<br>
    /// **Note:** The array must be uniquely held when mutating it.
    #[inline]
    pub unsafe fn uget_mut(&mut self, index: D) -> &mut A
        where S: DataMut
    {
        debug_assert!(self.data.is_unique());
        arraytraits::debug_bounds_check(self, &index);
        let off = Dimension::stride_offset(&index, &self.strides);
        &mut *self.ptr.offset(off)
    }

    /// Along `axis`, select the subview `index` and return a
    /// view with that axis removed.
    ///
    /// See [*Subviews*](#subviews) for full documentation.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    ///
    /// ```
    /// use ndarray::{arr2, ArrayView, Axis};
    ///
    /// let a = arr2(&[[1., 2.],    // -- axis 0, row 0
    ///                [3., 4.],    // -- axis 0, row 1
    ///                [5., 6.]]);  // -- axis 0, row 2
    /// //               \   \
    /// //                \   axis 1, column 1
    /// //                 axis 1, column 0
    /// assert!(
    ///     a.subview(Axis(0), 1) == ArrayView::from(&[3., 4.]) &&
    ///     a.subview(Axis(1), 1) == ArrayView::from(&[2., 4., 6.])
    /// );
    /// ```
    pub fn subview(&self, axis: Axis, index: Ix)
        -> ArrayView<A, <D as RemoveAxis>::Smaller>
        where D: RemoveAxis,
    {
        self.view().into_subview(axis, index)
    }

    /// Along `axis`, select the subview `index` and return a read-write view
    /// with the axis removed.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    ///
    /// ```
    /// use ndarray::{arr2, aview2, Axis};
    ///
    /// let mut a = arr2(&[[1., 2.],
    ///                    [3., 4.]]);
    ///
    /// a.subview_mut(Axis(1), 1).iadd_scalar(&10.);
    ///
    /// assert!(
    ///     a == aview2(&[[1., 12.],
    ///                   [3., 14.]])
    /// );
    /// ```
    pub fn subview_mut(&mut self, axis: Axis, index: Ix)
        -> ArrayViewMut<A, D::Smaller>
        where S: DataMut,
              D: RemoveAxis,
    {
        self.view_mut().into_subview(axis, index)
    }

    /// Collapse dimension `axis` into length one,
    /// and select the subview of `index` along that axis.
    ///
    /// **Panics** if `index` is past the length of the axis.
    pub fn isubview(&mut self, axis: Axis, index: Ix) {
        dimension::do_sub(&mut self.dim, &mut self.ptr, &self.strides,
                          axis.axis(), index)
    }

    /// Along `axis`, select the subview `index` and return `self`
    /// with that axis removed.
    ///
    /// See [`.subview()`](#method.subview) and [*Subviews*](#subviews) for full documentation.
    pub fn into_subview(mut self, axis: Axis, index: Ix)
        -> ArrayBase<S, <D as RemoveAxis>::Smaller>
        where D: RemoveAxis,
    {
        self.isubview(axis, index);
        // don't use reshape -- we always know it will fit the size,
        // and we can use remove_axis on the strides as well
        ArrayBase {
            data: self.data,
            ptr: self.ptr,
            dim: self.dim.remove_axis(axis),
            strides: self.strides.remove_axis(axis),
        }
    }

    /// Return an iterator that traverses over all dimensions but the innermost,
    /// and yields each inner row.
    ///
    /// For example, in a 2 × 2 × 3 array, the iterator element
    /// is a row of 3 elements (and there are 2 × 2 = 4 rows in total).
    ///
    /// Iterator element is `ArrayView<A, Ix>` (1D array view).
    ///
    /// ```
    /// use ndarray::arr3;
    /// let a = arr3(&[[[ 0,  1,  2],    // -- row 0, 0
    ///                 [ 3,  4,  5]],   // -- row 0, 1
    ///                [[ 6,  7,  8],    // -- row 1, 0
    ///                 [ 9, 10, 11]]]); // -- row 1, 1
    /// // `inner_iter` yields the four inner rows of the 3D array.
    /// let mut row_sums = a.inner_iter().map(|v| v.scalar_sum());
    /// assert_eq!(row_sums.collect::<Vec<_>>(), vec![3, 12, 21, 30]);
    /// ```
    pub fn inner_iter(&self) -> InnerIter<A, D> {
        iterators::new_inner_iter(self.view())
    }

    /// Return an iterator that traverses over all dimensions but the innermost,
    /// and yields each inner row.
    ///
    /// Iterator element is `ArrayViewMut<A, Ix>` (1D read-write array view).
    pub fn inner_iter_mut(&mut self) -> InnerIterMut<A, D>
        where S: DataMut
    {
        iterators::new_inner_iter_mut(self.view_mut())
    }

    /// Return an iterator that traverses over the outermost dimension
    /// and yields each subview.
    ///
    /// For example, in a 2 × 2 × 3 array, the iterator element
    /// is a 2 × 3 subview (and there are 2 in total).
    ///
    /// Iterator element is `ArrayView<A, D::Smaller>` (read-only array view).
    ///
    /// ```
    /// use ndarray::{arr3, Axis};
    ///
    /// let a = arr3(&[[[ 0,  1,  2],    // \ axis 0, submatrix 0
    ///                 [ 3,  4,  5]],   // /
    ///                [[ 6,  7,  8],    // \ axis 0, submatrix 1
    ///                 [ 9, 10, 11]]]); // /
    /// // `outer_iter` yields the two submatrices along axis 0.
    /// let mut iter = a.outer_iter();
    /// assert_eq!(iter.next().unwrap(), a.subview(Axis(0), 0));
    /// assert_eq!(iter.next().unwrap(), a.subview(Axis(0), 1));
    /// ```
    #[allow(deprecated)]
    pub fn outer_iter(&self) -> OuterIter<A, D::Smaller>
        where D: RemoveAxis,
    {
        self.view().into_outer_iter()
    }

    /// Return an iterator that traverses over the outermost dimension
    /// and yields each subview.
    ///
    /// Iterator element is `ArrayViewMut<A, D::Smaller>` (read-write array view).
    #[allow(deprecated)]
    pub fn outer_iter_mut(&mut self) -> OuterIterMut<A, D::Smaller>
        where S: DataMut,
              D: RemoveAxis,
    {
        self.view_mut().into_outer_iter()
    }

    /// Return an iterator that traverses over `axis`
    /// and yields each subview along it.
    ///
    /// For example, in a 2 × 2 × 3 array, with `axis` equal to 1,
    /// the iterator element
    /// is a 2 × 3 subview (and there are 2 in total).
    ///
    /// Iterator element is `ArrayView<A, D::Smaller>` (read-only array view).
    ///
    /// See [*Subviews*](#subviews) for full documentation.
    ///
    /// **Panics** if `axis` is out of bounds.
    pub fn axis_iter(&self, axis: Axis) -> OuterIter<A, D::Smaller>
        where D: RemoveAxis,
    {
        iterators::new_axis_iter(self.view(), axis.axis())
    }


    /// Return an iterator that traverses over `axis`
    /// and yields each mutable subview along it.
    ///
    /// Iterator element is `ArrayViewMut<A, D::Smaller>`
    /// (read-write array view).
    ///
    /// **Panics** if `axis` is out of bounds.
    pub fn axis_iter_mut(&mut self, axis: Axis) -> OuterIterMut<A, D::Smaller>
        where S: DataMut,
              D: RemoveAxis,
    {
        iterators::new_axis_iter_mut(self.view_mut(), axis.axis())
    }


    /// Return an iterator that traverses over `axis` by chunks of `size`,
    /// yielding non-overlapping views along that axis.
    ///
    /// Iterator element is `ArrayView<A, D>`
    ///
    /// The last view may have less elements if `size` does not divide
    /// the axis' dimension.
    ///
    /// **Panics** if `axis` is out of bounds.
    ///
    /// ```
    /// use ndarray::OwnedArray;
    /// use ndarray::{arr3, Axis};
    ///
    /// let a = OwnedArray::from_iter(0..28).into_shape((2, 7, 2)).unwrap();
    /// let mut iter = a.axis_chunks_iter(Axis(1), 2);
    ///
    /// // first iteration yields a 2 × 2 × 2 view
    /// assert_eq!(iter.next().unwrap(),
    ///            arr3(&[[[ 0,  1], [ 2, 3]],
    ///                   [[14, 15], [16, 17]]]));
    ///
    /// // however the last element is a 2 × 1 × 2 view since 7 % 2 == 1
    /// assert_eq!(iter.next_back().unwrap(), arr3(&[[[12, 13]],
    ///                                              [[26, 27]]]));
    /// ```
    pub fn axis_chunks_iter(&self, axis: Axis, size: usize) -> AxisChunksIter<A, D> {
        iterators::new_chunk_iter(self.view(), axis.axis(), size)
    }

    /// Return an iterator that traverses over `axis` by chunks of `size`,
    /// yielding non-overlapping read-write views along that axis.
    ///
    /// Iterator element is `ArrayViewMut<A, D>`
    ///
    /// **Panics** if `axis` is out of bounds.
    pub fn axis_chunks_iter_mut(&mut self, axis: Axis, size: usize)
        -> AxisChunksIterMut<A, D>
        where S: DataMut
    {
        iterators::new_chunk_iter_mut(self.view_mut(), axis.axis(), size)
    }

    // Return (length, stride) for diagonal
    fn diag_params(&self) -> (Ix, Ixs) {
        /* empty shape has len 1 */
        let len = self.dim.slice().iter().cloned().min().unwrap_or(1);
        let stride = self.strides()
                         .iter()
                         .fold(0, |sum, s| sum + s);
        (len, stride)
    }

    /// Return an view of the diagonal elements of the array.
    ///
    /// The diagonal is simply the sequence indexed by *(0, 0, .., 0)*,
    /// *(1, 1, ..., 1)* etc as long as all axes have elements.
    pub fn diag(&self) -> ArrayView<A, Ix> {
        self.view().into_diag()
    }

    /// Return a read-write view over the diagonal elements of the array.
    pub fn diag_mut(&mut self) -> ArrayViewMut<A, Ix>
        where S: DataMut,
    {
        self.view_mut().into_diag()
    }

    /// Return the diagonal as a one-dimensional array.
    pub fn into_diag(self) -> ArrayBase<S, Ix> {
        let (len, stride) = self.diag_params();
        ArrayBase {
            data: self.data,
            ptr: self.ptr,
            dim: len,
            strides: stride as Ix,
        }
    }

    /// Make the array unshared.
    ///
    /// This method is mostly only useful with unsafe code.
    fn ensure_unique(&mut self)
        where S: DataMut
    {
        debug_assert!(self.pointer_is_inbounds());
        S::ensure_unique(self);
        debug_assert!(self.pointer_is_inbounds());
    }

    /// Return `true` if the array data is laid out in contiguous “C order” in
    /// memory (where the last index is the most rapidly varying).
    ///
    /// Return `false` otherwise, i.e the array is possibly not
    /// contiguous in memory, it has custom strides, etc.
    pub fn is_standard_layout(&self) -> bool {
        let defaults = self.dim.default_strides();
        if self.strides == defaults {
            return true;
        }
        // check all dimensions -- a dimension of length 1 can have unequal strides
        for (&dim, (&s, &ds)) in zipsl(self.dim.slice(),
                                       zipsl(self.strides(), defaults.slice()))
        {
            if dim != 1 && s != (ds as Ixs) {
                return false;
            }
        }
        true
    }

    /// Return the array’s data as a slice, if it is contiguous and
    /// the element order corresponds to the memory order. Return `None` otherwise.
    pub fn as_slice(&self) -> Option<&[A]> {
        if self.is_standard_layout() {
            unsafe {
                Some(slice::from_raw_parts(self.ptr, self.len()))
            }
        } else {
            None
        }
    }

    /// Return the array’s data as a slice, if it is contiguous and
    /// the element order corresponds to the memory order. Return `None` otherwise.
    pub fn as_slice_mut(&mut self) -> Option<&mut [A]>
        where S: DataMut
    {
        if self.is_standard_layout() {
            self.ensure_unique();
            unsafe {
                Some(slice::from_raw_parts_mut(self.ptr, self.len()))
            }
        } else {
            None
        }
    }

    /// Transform the array into `shape`; any shape with the same number of
    /// elements is accepted.
    ///
    /// May clone all elements if needed to arrange elements in standard
    /// layout (and break sharing).
    ///
    /// **Panics** if shapes are incompatible.
    ///
    /// ```
    /// use ndarray::{rcarr1, rcarr2};
    ///
    /// assert!(
    ///     rcarr1(&[1., 2., 3., 4.]).reshape((2, 2))
    ///     == rcarr2(&[[1., 2.],
    ///                 [3., 4.]])
    /// );
    /// ```
    pub fn reshape<E>(&self, shape: E) -> ArrayBase<S, E>
        where S: DataShared + DataOwned,
              A: Clone,
              E: Dimension,
    {
        if shape.size_checked() != Some(self.dim.size()) {
            panic!("ndarray: incompatible shapes in reshape, attempted from: {:?}, to: {:?}",
                   self.dim.slice(),
                   shape.slice())
        }
        // Check if contiguous, if not => copy all, else just adapt strides
        if self.is_standard_layout() {
            let cl = self.clone();
            ArrayBase {
                data: cl.data,
                ptr: cl.ptr,
                strides: shape.default_strides(),
                dim: shape,
            }
        } else {
            let v = self.iter().map(|x| x.clone()).collect::<Vec<A>>();
            unsafe {
                ArrayBase::from_vec_dim_unchecked(shape, v)
            }
        }
    }

    /// Transform the array into `shape`; any shape with the same number of
    /// elements is accepted, but the source array or view must be
    /// contiguous, otherwise we cannot rearrange the dimension.
    ///
    /// **Errors** if the shapes don't have the same number of elements.<br>
    /// **Errors** if the input array is not c-contiguous (this will be
    /// slightly improved in the future).
    ///
    /// ```
    /// use ndarray::{aview1, aview2};
    ///
    /// assert!(
    ///     aview1(&[1., 2., 3., 4.]).into_shape((2, 2)).unwrap()
    ///     == aview2(&[[1., 2.],
    ///                 [3., 4.]])
    /// );
    /// ```
    pub fn into_shape<E>(self, shape: E) -> Result<ArrayBase<S, E>, ShapeError>
        where E: Dimension
    {
        if shape.size_checked() != Some(self.dim.size()) {
            return Err(error::incompatible_shapes(&self.dim, &shape));
        }
        // Check if contiguous, if not => copy all, else just adapt strides
        if self.is_standard_layout() {
            Ok(ArrayBase {
                data: self.data,
                ptr: self.ptr,
                strides: shape.default_strides(),
                dim: shape,
            })
        } else {
            Err(error::from_kind(error::ErrorKind::IncompatibleLayout))
        }
    }

    /// Act like a larger size and/or shape array by *broadcasting*
    /// into a larger shape, if possible.
    ///
    /// Return `None` if shapes can not be broadcast together.
    ///
    /// ***Background***
    ///
    ///  * Two axes are compatible if they are equal, or one of them is 1.
    ///  * In this instance, only the axes of the smaller side (self) can be 1.
    ///
    /// Compare axes beginning with the *last* axis of each shape.
    ///
    /// For example (1, 2, 4) can be broadcast into (7, 6, 2, 4)
    /// because its axes are either equal or 1 (or missing);
    /// while (2, 2) can *not* be broadcast into (2, 4).
    ///
    /// The implementation creates a view with strides set to zero for the
    /// axes that are to be repeated.
    ///
    /// The broadcasting documentation for Numpy has more information.
    ///
    /// ```
    /// use ndarray::{aview1, aview2};
    ///
    /// assert!(
    ///     aview1(&[1., 0.]).broadcast((10, 2)).unwrap()
    ///     == aview2(&[[1., 0.]; 10])
    /// );
    /// ```
    pub fn broadcast<E>(&self, dim: E) -> Option<ArrayView<A, E>>
        where E: Dimension
    {
        /// Return new stride when trying to grow `from` into shape `to`
        ///
        /// Broadcasting works by returning a "fake stride" where elements
        /// to repeat are in axes with 0 stride, so that several indexes point
        /// to the same element.
        ///
        /// **Note:** Cannot be used for mutable iterators, since repeating
        /// elements would create aliasing pointers.
        fn upcast<D: Dimension, E: Dimension>(to: &D, from: &E, stride: &E) -> Option<D> {
            let mut new_stride = to.clone();
            // begin at the back (the least significant dimension)
            // size of the axis has to either agree or `from` has to be 1
            if to.ndim() < from.ndim() {
                return None;
            }

            {
                let mut new_stride_iter = new_stride.slice_mut().iter_mut().rev();
                for ((er, es), dr) in from.slice().iter().rev()
                                        .zip(stride.slice().iter().rev())
                                        .zip(new_stride_iter.by_ref())
                {
                    /* update strides */
                    if *dr == *er {
                        /* keep stride */
                        *dr = *es;
                    } else if *er == 1 {
                        /* dead dimension, zero stride */
                        *dr = 0
                    } else {
                        return None;
                    }
                }

                /* set remaining strides to zero */
                for dr in new_stride_iter {
                    *dr = 0;
                }
            }
            Some(new_stride)
        }

        // Note: zero strides are safe precisely because we return an read-only view
        let broadcast_strides = match upcast(&dim, &self.dim, &self.strides) {
            Some(st) => st,
            None => return None,
        };
        unsafe { Some(ArrayView::new_(self.ptr, dim, broadcast_strides)) }
    }

    #[inline]
    fn broadcast_unwrap<E>(&self, dim: E) -> ArrayView<A, E>
        where E: Dimension,
    {
        #[cold]
        #[inline(never)]
        fn broadcast_panic<D, E>(from: &D, to: &E) -> !
            where D: Dimension,
                  E: Dimension,
        {
            panic!("ndarray: could not broadcast array from shape: {:?} to: {:?}",
                   from.slice(), to.slice())
        }

        match self.broadcast(dim.clone()) {
            Some(it) => it,
            None => broadcast_panic(&self.dim, &dim),
        }
    }

    /// Swap axes `ax` and `bx`.
    ///
    /// This does not move any data, it just adjusts the array’s dimensions
    /// and strides.
    ///
    /// **Panics** if the axes are out of bounds.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let mut a = arr2(&[[1., 2., 3.]]);
    /// a.swap_axes(0, 1);
    /// assert!(
    ///     a == arr2(&[[1.], [2.], [3.]])
    /// );
    /// ```
    pub fn swap_axes(&mut self, ax: usize, bx: usize) {
        self.dim.slice_mut().swap(ax, bx);
        self.strides.slice_mut().swap(ax, bx);
    }

    /// Transpose the array by reversing axes.
    ///
    /// Transposition reverses the order of the axes (dimensions and strides)
    /// while retaining the same data.
    pub fn reversed_axes(mut self) -> ArrayBase<S, D> {
        self.dim.slice_mut().reverse();
        self.strides.slice_mut().reverse();
        self
    }


    /// Return a slice of the array’s backing data in memory order.
    ///
    /// **Note:** Data memory order may not correspond to the index order
    /// of the array. Neither is the raw data slice is restricted to just the
    /// array’s view.<br>
    pub fn raw_data(&self) -> &[A]
        where S: DataOwned,
    {
        self.data.slice()
    }

    /// Return a mutable slice of the array’s backing data in memory order.
    ///
    /// **Note:** Data memory order may not correspond to the index order
    /// of the array. Neither is the raw data slice is restricted to just the
    /// array’s view.<br>
    ///
    /// **Note:** The data is uniquely held and nonaliased
    /// while it is mutably borrowed.
    pub fn raw_data_mut(&mut self) -> &mut [A]
        where S: DataOwned + DataMut,
    {
        self.ensure_unique();
        self.data.slice_mut()
    }

    fn pointer_is_inbounds(&self) -> bool {
        let slc = self.data.slice();
        if slc.is_empty() {
            // special case for data-less views
            return true;
        }
        let ptr = slc.as_ptr() as *mut _;
        let end =  unsafe {
            ptr.offset(slc.len() as isize)
        };
        self.ptr >= ptr && self.ptr <= end
    }

    /// Perform an elementwise assigment to `self` from `rhs`.
    ///
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting isn’t possible.
    pub fn assign<E: Dimension, S2>(&mut self, rhs: &ArrayBase<S2, E>)
        where S: DataMut,
              A: Clone,
              S2: Data<Elem=A>,
    {
        self.zip_mut_with(rhs, |x, y| *x = y.clone());
    }

    /// Perform an elementwise assigment to `self` from scalar `x`.
    pub fn assign_scalar(&mut self, x: &A)
        where S: DataMut, A: Clone,
    {
        self.unordered_foreach_mut(move |elt| *elt = x.clone());
    }

    fn zip_with_mut_same_shape<B, S2, E, F>(&mut self, rhs: &ArrayBase<S2, E>, mut f: F)
        where S: DataMut,
              S2: Data<Elem=B>,
              E: Dimension,
              F: FnMut(&mut A, &B)
    {
        debug_assert_eq!(self.shape(), rhs.shape());
        if let Some(self_s) = self.as_slice_mut() {
            if let Some(rhs_s) = rhs.as_slice() {
                let len = cmp::min(self_s.len(), rhs_s.len());
                let s = &mut self_s[..len];
                let r = &rhs_s[..len];
                for i in 0..len {
                    f(&mut s[i], &r[i]);
                }
                return;
            }
        }
        // otherwise, fall back to the outer iter
        self.zip_with_mut_outer_iter(rhs, f);
    }

    #[inline(always)]
    fn zip_with_mut_outer_iter<B, S2, E, F>(&mut self, rhs: &ArrayBase<S2, E>, mut f: F)
        where S: DataMut,
              S2: Data<Elem=B>,
              E: Dimension,
              F: FnMut(&mut A, &B)
    {
        debug_assert_eq!(self.shape(), rhs.shape());
        // otherwise, fall back to the outer iter
        let mut try_slices = true;
        let mut rows = self.inner_iter_mut().zip(rhs.inner_iter());
        for (mut s_row, r_row) in &mut rows {
            if try_slices {
                if let Some(self_s) = s_row.as_slice_mut() {
                    if let Some(rhs_s) = r_row.as_slice() {
                        let len = cmp::min(self_s.len(), rhs_s.len());
                        let s = &mut self_s[..len];
                        let r = &rhs_s[..len];
                        for i in 0..len {
                            f(&mut s[i], &r[i]);
                        }
                        continue;
                    }
                }
                try_slices = false;
            }
            unsafe {
                for i in 0..s_row.len() {
                    f(s_row.uget_mut(i), r_row.uget(i))
                }
            }
        }
    }

    fn zip_mut_with_elem<B, F>(&mut self, rhs_elem: &B, mut f: F)
        where S: DataMut,
              F: FnMut(&mut A, &B)
    {
        self.unordered_foreach_mut(move |elt| f(elt, rhs_elem));
    }

    // FIXME: Guarantee the order here or not?
    /// Traverse two arrays in unspecified order, in lock step,
    /// calling the closure `f` on each element pair.
    ///
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting isn’t possible.
    #[inline]
    pub fn zip_mut_with<B, S2, E, F>(&mut self, rhs: &ArrayBase<S2, E>, f: F)
        where S: DataMut,
              S2: Data<Elem=B>,
              E: Dimension,
              F: FnMut(&mut A, &B)
    {
        if rhs.dim.ndim() == 0 {
            // Skip broadcast from 0-dim array
            unsafe {
                let rhs_elem = &*rhs.ptr;
                self.zip_mut_with_elem(rhs_elem, f);
            }
        } else if self.dim.ndim() == rhs.dim.ndim() && self.shape() == rhs.shape() {
            self.zip_with_mut_same_shape(rhs, f);
        } else {
            let rhs_broadcast = rhs.broadcast_unwrap(self.dim());
            self.zip_with_mut_outer_iter(&rhs_broadcast, f);
        }
    }

    /// Traverse the array elements in order and apply a fold,
    /// returning the resulting value.
    pub fn fold<'a, F, B>(&'a self, mut init: B, mut f: F) -> B
        where F: FnMut(B, &'a A) -> B, A: 'a
    {
        if let Some(slc) = self.as_slice() {
            for elt in slc {
                init = f(init, elt);
            }
            return init;
        }
        for row in self.inner_iter() {
            for elt in row {
                init = f(init, elt);
            }
        }
        init
    }

    /// Apply `f` elementwise and return a new array with
    /// the results.
    ///
    /// Return an array with the same shape as *self*.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[ 0., 1.],
    ///                [-1., 2.]]);
    /// assert!(
    ///     a.map(|x| *x >= 1.0)
    ///     == arr2(&[[false, true],
    ///               [false, true]])
    /// );
    /// ```
    pub fn map<'a, B, F>(&'a self, mut f: F) -> OwnedArray<B, D>
        where F: FnMut(&'a A) -> B,
              A: 'a,
    {
        // Use an `unsafe` block to do this efficiently.
        // We know that iter will produce exactly .size() elements,
        // and the loop can vectorize if it's clean (without branch
        // to grow the vector).
        let mut res = Vec::with_capacity(self.dim.size());
        let mut out_ptr = res.as_mut_ptr();
        let mut len = 0;
        for elt in self.iter() {
            unsafe {
                ptr::write(out_ptr, f(elt));
                len += 1;
                res.set_len(len);
                out_ptr = out_ptr.offset(1);
            }
        }
        unsafe {
            ArrayBase::from_vec_dim_unchecked(self.dim.clone(), res)
        }
    }
}
