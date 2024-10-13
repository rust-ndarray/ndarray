use crate::{iter::Axes, ArrayBase, Axis, AxisDescription, Dimension, LayoutRef, RawData, Slice, SliceArg};

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
    /// - if [`SliceInfoElem::NewAxis`] is in `info`, e.g. if [`NewAxis`] was
    ///   used in the [`s!`] macro
    /// - if `D` is `IxDyn` and `info` does not match the number of array axes
    #[track_caller]
    pub fn slice_collapse<I>(&mut self, info: I)
    where I: SliceArg<D>
    {
        self.as_mut().slice_collapse(info);
    }

    /// Slice the array in place along the specified axis.
    ///
    /// **Panics** if an index is out of bounds or step size is zero.<br>
    /// **Panics** if `axis` is out of bounds.
    #[track_caller]
    pub fn slice_axis_inplace(&mut self, axis: Axis, indices: Slice)
    {
        self.as_mut().slice_axis_inplace(axis, indices);
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
        self.as_mut().slice_each_axis_inplace(f);
    }

    /// Selects `index` along the axis, collapsing the axis into length one.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    #[track_caller]
    pub fn collapse_axis(&mut self, axis: Axis, index: usize)
    {
        self.as_mut().collapse_axis(axis, index);
    }

    /// Return `true` if the array data is laid out in contiguous “C order” in
    /// memory (where the last index is the most rapidly varying).
    ///
    /// Return `false` otherwise, i.e. the array is possibly not
    /// contiguous in memory, it has custom strides, etc.
    pub fn is_standard_layout(&self) -> bool
    {
        self.as_ref().is_standard_layout()
    }

    /// Return true if the array is known to be contiguous.
    pub(crate) fn is_contiguous(&self) -> bool
    {
        self.as_ref().is_contiguous()
    }

    /// Return an iterator over the length and stride of each axis.
    pub fn axes(&self) -> Axes<'_, D>
    {
        self.as_ref().axes()
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
        LayoutRef::max_stride_axis(&self.as_ref())
    }

    /// Reverse the stride of `axis`.
    ///
    /// ***Panics*** if the axis is out of bounds.
    #[track_caller]
    pub fn invert_axis(&mut self, axis: Axis)
    {
        self.as_mut().invert_axis(axis);
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
        self.as_mut().swap_axes(ax, bx);
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
        self.as_mut().merge_axes(take, into)
    }

    /// Return the strides of the array as a slice.
    pub fn strides(&self) -> &[isize]
    {
        self.as_ref().strides()
    }
}
