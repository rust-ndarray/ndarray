use crate::{
    iter::Axes,
    ArrayBase,
    Axis,
    AxisDescription,
    Dimension,
    NdIndex,
    RawArrayView,
    RawData,
    RawDataMut,
    Slice,
    SliceArg,
};

/// Functions coming from RawRef
impl<A, S: RawData<Elem = A>, D: Dimension> ArrayBase<S, D>
{
    /// Return a raw pointer to the element at `index`, or return `None`
    /// if the index is out of bounds.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[1., 2.], [3., 4.]]);
    ///
    /// let v = a.raw_view();
    /// let p = a.get_ptr((0, 1)).unwrap();
    ///
    /// assert_eq!(unsafe { *p }, 2.);
    /// ```
    pub fn get_ptr<I>(&self, index: I) -> Option<*const A>
    where I: NdIndex<D>
    {
        self.as_raw_ref().get_ptr(index)
    }

    /// Return a raw pointer to the element at `index`, or return `None`
    /// if the index is out of bounds.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let mut a = arr2(&[[1., 2.], [3., 4.]]);
    ///
    /// let v = a.raw_view_mut();
    /// let p = a.get_mut_ptr((0, 1)).unwrap();
    ///
    /// unsafe {
    ///     *p = 5.;
    /// }
    ///
    /// assert_eq!(a.get((0, 1)), Some(&5.));
    /// ```
    pub fn get_mut_ptr<I>(&mut self, index: I) -> Option<*mut A>
    where
        S: RawDataMut<Elem = A>,
        I: NdIndex<D>,
    {
        self.as_raw_ref_mut().get_mut_ptr(index)
    }

    /// Return a pointer to the first element in the array.
    ///
    /// Raw access to array elements needs to follow the strided indexing
    /// scheme: an element at multi-index *I* in an array with strides *S* is
    /// located at offset
    ///
    /// *Σ<sub>0 ≤ k < d</sub> I<sub>k</sub> × S<sub>k</sub>*
    ///
    /// where *d* is `self.ndim()`.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const A
    {
        self.as_raw_ref().as_ptr()
    }

    /// Return a raw view of the array.
    #[inline]
    pub fn raw_view(&self) -> RawArrayView<S::Elem, D>
    {
        self.as_raw_ref().raw_view()
    }
}

/// Functions coming from LayoutRef
impl<S: RawData, D: Dimension> ArrayBase<S, D>
{
    /// Slice the array in place without changing the number of dimensions.
    ///
    /// In particular, if an axis is sliced with an index, the axis is
    /// collapsed, as in [`.collapse_axis()`], rather than removed, as in
    /// [`.slice_move()`] or [`.index_axis_move()`].
    ///
    /// [`.collapse_axis()`]: Self::collapse_axis
    /// [`.slice_move()`]: Self::slice_move
    /// [`.index_axis_move()`]: Self::index_axis_move
    ///
    /// See [*Slicing*](#slicing) for full documentation.
    /// See also [`s!`], [`SliceArg`], and [`SliceInfo`](crate::SliceInfo).
    ///
    /// **Panics** in the following cases:
    ///
    /// - if an index is out of bounds
    /// - if a step size is zero
    /// - if [`NewAxis`](`crate::SliceInfoElem::NewAxis`) is in `info`, e.g. if `NewAxis` was
    ///   used in the [`s!`] macro
    /// - if `D` is `IxDyn` and `info` does not match the number of array axes
    #[track_caller]
    pub fn slice_collapse<I>(&mut self, info: I)
    where I: SliceArg<D>
    {
        self.as_layout_ref_mut().slice_collapse(info);
    }

    /// Slice the array in place along the specified axis.
    ///
    /// **Panics** if an index is out of bounds or step size is zero.<br>
    /// **Panics** if `axis` is out of bounds.
    #[track_caller]
    pub fn slice_axis_inplace(&mut self, axis: Axis, indices: Slice)
    {
        self.as_layout_ref_mut().slice_axis_inplace(axis, indices);
    }

    /// Slice the array in place, with a closure specifying the slice for each
    /// axis.
    ///
    /// This is especially useful for code which is generic over the
    /// dimensionality of the array.
    ///
    /// **Panics** if an index is out of bounds or step size is zero.
    #[track_caller]
    pub fn slice_each_axis_inplace<F>(&mut self, f: F)
    where F: FnMut(AxisDescription) -> Slice
    {
        self.as_layout_ref_mut().slice_each_axis_inplace(f);
    }

    /// Selects `index` along the axis, collapsing the axis into length one.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    #[track_caller]
    pub fn collapse_axis(&mut self, axis: Axis, index: usize)
    {
        self.as_layout_ref_mut().collapse_axis(axis, index);
    }

    /// Return `true` if the array data is laid out in contiguous “C order” in
    /// memory (where the last index is the most rapidly varying).
    ///
    /// Return `false` otherwise, i.e. the array is possibly not
    /// contiguous in memory, it has custom strides, etc.
    pub fn is_standard_layout(&self) -> bool
    {
        self.as_layout_ref().is_standard_layout()
    }

    /// Return true if the array is known to be contiguous.
    pub(crate) fn is_contiguous(&self) -> bool
    {
        self.as_layout_ref().is_contiguous()
    }

    /// Return an iterator over the length and stride of each axis.
    pub fn axes(&self) -> Axes<'_, D>
    {
        self.as_layout_ref().axes()
    }

    /*
    /// Return the axis with the least stride (by absolute value)
    pub fn min_stride_axis(&self) -> Axis {
        self.dim.min_stride_axis(&self.strides)
    }
    */

    /// Return the axis with the greatest stride (by absolute value),
    /// preferring axes with len > 1.
    pub fn max_stride_axis(&self) -> Axis
    {
        self.as_layout_ref().max_stride_axis()
    }

    /// Reverse the stride of `axis`.
    ///
    /// ***Panics*** if the axis is out of bounds.
    #[track_caller]
    pub fn invert_axis(&mut self, axis: Axis)
    {
        self.as_layout_ref_mut().invert_axis(axis);
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
    #[track_caller]
    pub fn swap_axes(&mut self, ax: usize, bx: usize)
    {
        self.as_layout_ref_mut().swap_axes(ax, bx);
    }

    /// If possible, merge in the axis `take` to `into`.
    ///
    /// Returns `true` iff the axes are now merged.
    ///
    /// This method merges the axes if movement along the two original axes
    /// (moving fastest along the `into` axis) can be equivalently represented
    /// as movement along one (merged) axis. Merging the axes preserves this
    /// order in the merged axis. If `take` and `into` are the same axis, then
    /// the axis is "merged" if its length is ≤ 1.
    ///
    /// If the return value is `true`, then the following hold:
    ///
    /// * The new length of the `into` axis is the product of the original
    ///   lengths of the two axes.
    ///
    /// * The new length of the `take` axis is 0 if the product of the original
    ///   lengths of the two axes is 0, and 1 otherwise.
    ///
    /// If the return value is `false`, then merging is not possible, and the
    /// original shape and strides have been preserved.
    ///
    /// Note that the ordering constraint means that if it's possible to merge
    /// `take` into `into`, it's usually not possible to merge `into` into
    /// `take`, and vice versa.
    ///
    /// ```
    /// use ndarray::Array3;
    /// use ndarray::Axis;
    ///
    /// let mut a = Array3::<f64>::zeros((2, 3, 4));
    /// assert!(a.merge_axes(Axis(1), Axis(2)));
    /// assert_eq!(a.shape(), &[2, 1, 12]);
    /// ```
    ///
    /// ***Panics*** if an axis is out of bounds.
    #[track_caller]
    pub fn merge_axes(&mut self, take: Axis, into: Axis) -> bool
    {
        self.as_layout_ref_mut().merge_axes(take, into)
    }

    /// Return the total number of elements in the array.
    pub fn len(&self) -> usize
    {
        self.as_layout_ref().len()
    }

    /// Return the length of `axis`.
    ///
    /// The axis should be in the range `Axis(` 0 .. *n* `)` where *n* is the
    /// number of dimensions (axes) of the array.
    ///
    /// ***Panics*** if the axis is out of bounds.
    #[track_caller]
    pub fn len_of(&self, axis: Axis) -> usize
    {
        self.as_layout_ref().len_of(axis)
    }

    /// Return whether the array has any elements
    pub fn is_empty(&self) -> bool
    {
        self.as_layout_ref().is_empty()
    }

    /// Return the number of dimensions (axes) in the array
    pub fn ndim(&self) -> usize
    {
        self.as_layout_ref().ndim()
    }

    /// Return the shape of the array in its “pattern” form,
    /// an integer in the one-dimensional case, tuple in the n-dimensional cases
    /// and so on.
    pub fn dim(&self) -> D::Pattern
    {
        self.as_layout_ref().dim()
    }

    /// Return the shape of the array as it's stored in the array.
    ///
    /// This is primarily useful for passing to other `ArrayBase`
    /// functions, such as when creating another array of the same
    /// shape and dimensionality.
    ///
    /// ```
    /// use ndarray::Array;
    ///
    /// let a = Array::from_elem((2, 3), 5.);
    ///
    /// // Create an array of zeros that's the same shape and dimensionality as `a`.
    /// let b = Array::<f64, _>::zeros(a.raw_dim());
    /// ```
    pub fn raw_dim(&self) -> D
    {
        self.as_layout_ref().raw_dim()
    }

    /// Return the shape of the array as a slice.
    ///
    /// Note that you probably don't want to use this to create an array of the
    /// same shape as another array because creating an array with e.g.
    /// [`Array::zeros()`](ArrayBase::zeros) using a shape of type `&[usize]`
    /// results in a dynamic-dimensional array. If you want to create an array
    /// that has the same shape and dimensionality as another array, use
    /// [`.raw_dim()`](ArrayBase::raw_dim) instead:
    ///
    /// ```rust
    /// use ndarray::{Array, Array2};
    ///
    /// let a = Array2::<i32>::zeros((3, 4));
    /// let shape = a.shape();
    /// assert_eq!(shape, &[3, 4]);
    ///
    /// // Since `a.shape()` returned `&[usize]`, we get an `ArrayD` instance:
    /// let b = Array::zeros(shape);
    /// assert_eq!(a.clone().into_dyn(), b);
    ///
    /// // To get the same dimension type, use `.raw_dim()` instead:
    /// let c = Array::zeros(a.raw_dim());
    /// assert_eq!(a, c);
    /// ```
    pub fn shape(&self) -> &[usize]
    {
        self.as_layout_ref().shape()
    }

    /// Return the strides of the array as a slice.
    pub fn strides(&self) -> &[isize]
    {
        self.as_layout_ref().strides()
    }

    /// Return the stride of `axis`.
    ///
    /// The axis should be in the range `Axis(` 0 .. *n* `)` where *n* is the
    /// number of dimensions (axes) of the array.
    ///
    /// ***Panics*** if the axis is out of bounds.
    #[track_caller]
    pub fn stride_of(&self, axis: Axis) -> isize
    {
        self.as_layout_ref().stride_of(axis)
    }
}
