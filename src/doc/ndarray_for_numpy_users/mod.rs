//! `ndarray` for NumPy users.
//!
//! This is an introductory guide to `ndarray` for people with experience using
//! NumPy, although it may also be useful to others. For a more general
//! introduction to `ndarray`'s array type `ArrayBase`, see the [`ArrayBase`
//! docs][ArrayBase].
//!
//! # Contents
//!
//! * [Similarities](#similarities)
//! * [Some key differences](#some-key-differences)
//! * [Other Rust array/matrix crates](#other-rust-arraymatrix-crates)
//! * [Rough `ndarray`–NumPy equivalents](#rough-ndarraynumpy-equivalents)
//!
//!   * [Array creation](#array-creation)
//!   * [Indexing and slicing](#indexing-and-slicing)
//!   * [Shape and strides](#shape-and-strides)
//!   * [Mathematics](#mathematics)
//!   * [Array manipulation](#array-manipulation)
//!   * [Iteration](#iteration)
//!   * [Convenience methods for 2-D arrays](#convenience-methods-for-2-d-arrays)
//!
//! # Similarities
//!
//! `ndarray`'s array type ([`ArrayBase`][ArrayBase]), is very similar to
//! NumPy's array type (`numpy.ndarray`):
//!
//! * Arrays have a single element type.
//! * Arrays can have arbitrarily many dimensions.
//! * Arrays can have arbitrary strides.
//! * Indexing starts at zero, not one.
//! * The default memory layout is row-major, and the default iterators follow
//!   row-major order (also called "logical order" in the documentation).
//! * Arithmetic operators work elementwise. (For example, `a * b` performs
//!   elementwise multiplication, not matrix multiplication.)
//! * Owned arrays are contiguous in memory.
//! * Many operations, such as slicing, are very cheap because they can return
//!   a view of an array instead of copying the data.
//!
//! NumPy has many features that `ndarray` doesn't have yet, such as:
//!
//! * [index arrays](https://docs.scipy.org/doc/numpy/user/basics.indexing.html#index-arrays)
//! * [mask index arrays](https://docs.scipy.org/doc/numpy/user/basics.indexing.html#boolean-or-mask-index-arrays)
//! * co-broadcasting (`ndarray` only supports broadcasting the right-hand array in a binary operation.)
//!
//! # Some key differences
//!
//! <table>
//! <tr>
//! <th>
//!
//! NumPy
//!
//! </th>
//! <th>
//!
//! `ndarray`
//!
//! </th>
//! </tr>
//!
//! <tr>
//! <td>
//!
//! In NumPy, there is no distinction between owned arrays, views, and mutable
//! views. There can be multiple arrays (instances of `numpy.ndarray`) that
//! mutably reference the same data.
//!
//! </td>
//! <td>
//!
//! In `ndarray`, all arrays are instances of [`ArrayBase`][ArrayBase], but
//! `ArrayBase` is generic over the ownership of the data. [`Array`][Array]
//! owns its data; [`ArrayView`][ArrayView] is a view;
//! [`ArrayViewMut`][ArrayViewMut] is a mutable view; [`CowArray`][CowArray]
//! either owns its data or is a view (with copy-on-write mutation of the view
//! variant); and [`ArcArray`][ArcArray] has a reference-counted pointer to its
//! data (with copy-on-write mutation). Arrays and views follow Rust's aliasing
//! rules.
//!
//! </td>
//! </tr>
//!
//! <tr>
//! <td>
//!
//! In NumPy, all arrays are dynamic-dimensional.
//!
//! </td>
//! <td>
//!
//! In `ndarray`, you can create fixed-dimension arrays, such as
//! [`Array2`][Array2]. This takes advantage of the type system to help you
//! write correct code and also avoids small heap allocations for the shape and
//! strides.
//!
//! </td>
//! </tr>
//!
//! <tr>
//! <td>
//!
//! When slicing in NumPy, the indices are `start`, `start + step`, `start +
//! 2*step`, … until reaching `end` (exclusive).
//!
//! </td>
//! <td>
//!
//! When slicing in `ndarray`, the axis is first sliced with `start..end`. Then if
//! `step` is positive, the first index is the front of the slice; if `step` is
//! negative, the first index is the back of the slice. This means that the
//! behavior is the same as NumPy except when `step < -1`. See the docs for the
//! [`s![]` macro][s!] for more details.
//!
//! </td>
//! </tr>
//! </table>
//!
//! # Other Rust array/matrix crates
//!
//! Of the array/matrix types in Rust crates, the `ndarray` array type is probably
//! the most similar to NumPy's arrays and is the most flexible. However, if your
//! use-case is constrained to linear algebra on 1-D and 2-D vectors and matrices,
//! it might be worth considering other crates:
//!
//! * [`nalgebra`](https://crates.io/crates/nalgebra) provides 1-D and 2-D
//!   column-major vector and matrix types for linear algebra. Vectors and matrices
//!   can have constant or dynamic shapes, and `nalgebra` uses the type system to
//!   provide compile-time checking of shapes, not just the number of dimensions.
//!   `nalgebra` provides convenient functionality for geometry (e.g. coordinate
//!   transformations) and linear algebra.
//! * [`cgmath`](https://crates.io/crates/cgmath) provides 1-D and 2-D column-major
//!   types of shape 4×4 or smaller. It's primarily designed for computer graphics
//!   and provides convenient functionality for geometry (e.g. coordinate
//!   transformations). Similar to `nalgebra`, `cgmath` uses the type system to
//!   provide compile-time checking of shapes.
//! * [`rulinalg`](https://crates.io/crates/rulinalg) provides 1-D and 2-D
//!   row-major vector and matrix types with dynamic shapes. Similar to `ndarray`,
//!   `rulinalg` provides compile-time checking of the number of dimensions, but
//!   not shapes. `rulinalg` provides pure-Rust implementations of linear algebra
//!   operations.
//! * If there's another crate that should be listed here, please let us know.
//!
//! In contrast to these crates, `ndarray` provides an *n*-dimensional array type,
//! so it's not restricted to 1-D and 2-D vectors and matrices. Also, operators
//! operate elementwise by default, so the multiplication operator `*` performs
//! elementwise multiplication instead of matrix multiplication. (You have to
//! specifically call `.dot()` if you want matrix multiplication.)
//!
//! Linear algebra with `ndarray` is provided by another crates,
//! [`ndarray-linalg`](https://crates.io/crates/ndarray-linalg).
//!
//! # Rough `ndarray`–NumPy equivalents
//!
//! These tables provide some rough equivalents of NumPy operations in `ndarray`.
//! There are a variety of other methods that aren't included in these tables,
//! including shape-manipulation, array creation, and iteration routines.
//!
//! It's assumed that you've imported NumPy like this:
//!
//! ```python
//! import numpy as np
//! ```
//!
//! and `ndarray` like this:
//!
//! ```
//! extern crate ndarray;
//!
//! use ndarray::prelude::*;
//! #
//! # fn main() {}
//! ```
//!
//! ## Array creation
//!
//! This table contains ways to create arrays from scratch. For creating arrays by
//! operations on other arrays (e.g. arithmetic), see the other tables. Also see
//! the [`::from_vec()`][::from_vec()], [`::from_iter()`][::from_iter()],
//! [`::default()`][::default()], [`::from_shape_fn()`][::from_shape_fn()], and
//! [`::from_shape_vec_unchecked()`][::from_shape_vec_unchecked()] methods.
//!
//! NumPy | `ndarray` | Notes
//! ------|-----------|------
//! `np.array([[1.,2.,3.], [4.,5.,6.]])` | [`array![[1.,2.,3.], [4.,5.,6.]]`][array!] or [`arr2(&[[1.,2.,3.], [4.,5.,6.]])`][arr2()] | 2×3 floating-point array literal
//! `np.arange(0., 10., 0.5)` or `np.r_[:10.:0.5]` | [`Array::range(0., 10., 0.5)`][::range()] | create a 1-D array with values `0.`, `0.5`, …, `9.5`
//! `np.linspace(0., 10., 11)` or `np.r_[:10.:11j]` | [`Array::linspace(0., 10., 11)`][::linspace()] | create a 1-D array with 11 elements with values `0.`, …, `10.`
//! `np.ones((3, 4, 5))` | [`Array::ones((3, 4, 5))`][::ones()] | create a 3×4×5 array filled with ones (inferring the element type)
//! `np.zeros((3, 4, 5))` | [`Array::zeros((3, 4, 5))`][::zeros()] | create a 3×4×5 array filled with zeros (inferring the element type)
//! `np.zeros((3, 4, 5), order='F')` | [`Array::zeros((3, 4, 5).f())`][::zeros()] | create a 3×4×5 array with Fortran (column-major) memory layout filled with zeros (inferring the element type)
//! `np.zeros_like(a, order='C')` | [`Array::zeros(a.raw_dim())`][::zeros()] | create an array of zeros of the shape shape as `a`, with row-major memory layout (unlike NumPy, this infers the element type from context instead of duplicating `a`'s element type)
//! `np.full((3, 4), 7.)` | [`Array::from_elem((3, 4), 7.)`][::from_elem()] | create a 3×4 array filled with the value `7.`
//! `np.eye(3)` | [`Array::eye(3)`][::eye()] | create a 3×3 identity matrix (inferring the element type)
//! `np.array([1, 2, 3, 4]).reshape((2, 2))` | [`Array::from_shape_vec((2, 2), vec![1, 2, 3, 4])?`][::from_shape_vec()] | create a 2×2 array from the elements in the list/`Vec`
//! `np.array([1, 2, 3, 4]).reshape((2, 2), order='F')` | [`Array::from_shape_vec((2, 2).f(), vec![1, 2, 3, 4])?`][::from_shape_vec()] | create a 2×2 array from the elements in the list/`Vec` using Fortran (column-major) order
//! `np.random` | See the [`ndarray-rand`](https://crates.io/crates/ndarray-rand) crate. | create arrays of random numbers
//!
//! Note that the examples in the table rely on the compiler inferring the
//! element type and dimensionality from context, which is usually sufficient.
//! However, if the compiler cannot infer the types, you can specify them
//! manually. These are examples of creating a 3-D Fortran-layout array of
//! `f64`s:
//!
//! ```
//! # use ndarray::prelude::*;
//! #
//! // This is an example where the compiler can infer the element type
//! // because `f64::sin` can only be called on `f64` elements:
//! let arr1 = Array::zeros((3, 2, 4).f());
//! arr1.mapv(f64::sin);
//!
//! // Specify just the element type and infer the dimensionality:
//! let arr2 = Array::<f64, _>::zeros((3, 2, 4).f());
//! let arr3: Array<f64, _> = Array::zeros((3, 2, 4).f());
//!
//! // Specify both the element type and dimensionality:
//! let arr4 = Array3::<f64>::zeros((3, 2, 4).f());
//! let arr5: Array3<f64> = Array::zeros((3, 2, 4).f());
//! let arr6 = Array::<f64, Ix3>::zeros((3, 2, 4).f());
//! let arr7: Array<f64, Ix3> = Array::zeros((3, 2, 4).f());
//! ```
//!
//! ## Indexing and slicing
//!
//! A few notes:
//!
//! * Indices start at 0. For example, "row 1" is the second row in the array.
//!
//! * Some methods have multiple variants in terms of ownership and mutability.
//!   Only the non-mutable methods that take the array by reference are listed in
//!   this table. For example, [`.slice()`][.slice()] also has corresponding
//!   methods [`.slice_mut()`][.slice_mut()], [`.slice_move()`][.slice_move()], and
//!   [`.slice_collapse()`][.slice_collapse()].
//!
//! * The behavior of slicing is slightly different from NumPy for slices with
//!   `step < -1`. See the docs for the [`s![]` macro][s!] for more details.
//!
//! NumPy | `ndarray` | Notes
//! ------|-----------|------
//! `a[-1]` | [`a[a.len() - 1]`][.index()] | access the last element in 1-D array `a`
//! `a[1, 4]` | [`a[[1, 4]]`][.index()] | access the element in row 1, column 4
//! `a[1]` or `a[1, :, :]` | [`a.slice(s![1, .., ..])`][.slice()] or [`a.index_axis(Axis(0), 1)`][.index_axis()] | get a 2-D subview of a 3-D array at index 1 of axis 0
//! `a[0:5]` or `a[:5]` or `a[0:5, :]` | [`a.slice(s![0..5, ..])`][.slice()] or [`a.slice(s![..5, ..])`][.slice()] or [`a.slice_axis(Axis(0), Slice::from(0..5))`][.slice_axis()] | get the first 5 rows of a 2-D array
//! `a[-5:]` or `a[-5:, :]` | [`a.slice(s![-5.., ..])`][.slice()] or [`a.slice_axis(Axis(0), Slice::from(-5..))`][.slice_axis()] | get the last 5 rows of a 2-D array
//! `a[:3, 4:9]` | [`a.slice(s![..3, 4..9])`][.slice()] | columns 4, 5, 6, 7, and 8 of the first 3 rows
//! `a[1:4:2, ::-1]` | [`a.slice(s![1..4;2, ..;-1])`][.slice()] | rows 1 and 3 with the columns in reverse order
//!
//! ## Shape and strides
//!
//! Note that [`a.shape()`][.shape()], [`a.dim()`][.dim()], and
//! [`a.raw_dim()`][.raw_dim()] all return the shape of the array, but as
//! different types. `a.shape()` returns the shape as `&[Ix]`, (where
//! [`Ix`][Ix] is `usize`) which is useful for general operations on the shape.
//! `a.dim()` returns the shape as `D::Pattern`, which is useful for
//! pattern-matching shapes. `a.raw_dim()` returns the shape as `D`, which is
//! useful for creating other arrays of the same shape.
//!
//! NumPy | `ndarray` | Notes
//! ------|-----------|------
//! `np.ndim(a)` or `a.ndim` | [`a.ndim()`][.ndim()] | get the number of dimensions of array `a`
//! `np.size(a)` or `a.size` | [`a.len()`][.len()] | get the number of elements in array `a`
//! `np.shape(a)` or `a.shape` | [`a.shape()`][.shape()] or [`a.dim()`][.dim()] | get the shape of array `a`
//! `a.shape[axis]` | [`a.len_of(Axis(axis))`][.len_of()] | get the length of an axis
//! `a.strides` | [`a.strides()`][.strides()] | get the strides of array `a`
//! `np.size(a) == 0` or `a.size == 0` | [`a.is_empty()`][.is_empty()] | check if the array has zero elements
//!
//! ## Mathematics
//!
//! Note that [`.mapv()`][.mapv()] has corresponding methods [`.map()`][.map()],
//! [`.mapv_into()`][.mapv_into()], [`.map_inplace()`][.map_inplace()], and
//! [`.mapv_inplace()`][.mapv_inplace()]. Also look at [`.fold()`][.fold()],
//! [`.visit()`][.visit()], [`.fold_axis()`][.fold_axis()], and
//! [`.map_axis()`][.map_axis()].
//!
//! <table>
//! <tr><th>
//!
//! NumPy
//!
//! </th><th>
//!
//! `ndarray`
//!
//! </th><th>
//!
//! Notes
//!
//! </th></tr>
//!
//! <tr><td>
//!
//! `a.transpose()` or `a.T`
//!
//! </td><td>
//!
//! [`a.t()`][.t()] or [`a.reversed_axes()`][.reversed_axes()]
//!
//! </td><td>
//!
//! transpose of array `a` (view for `.t()` or by-move for `.reversed_axes()`)
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `mat1.dot(mat2)`
//!
//! </td><td>
//!
//! [`mat1.dot(&mat2)`][matrix-* dot]
//!
//! </td><td>
//!
//! 2-D matrix multiply
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `mat.dot(vec)`
//!
//! </td><td>
//!
//! [`mat.dot(&vec)`][matrix-* dot]
//!
//! </td><td>
//!
//! 2-D matrix dot 1-D column vector
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `vec.dot(mat)`
//!
//! </td><td>
//!
//! [`vec.dot(&mat)`][vec-* dot]
//!
//! </td><td>
//!
//! 1-D row vector dot 2-D matrix
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `vec1.dot(vec2)`
//!
//! </td><td>
//!
//! [`vec1.dot(&vec2)`][vec-* dot]
//!
//! </td><td>
//!
//! vector dot product
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `a * b`, `a + b`, etc.
//!
//! </td><td>
//!
//! [`a * b`, `a + b`, etc.](../../struct.ArrayBase.html#arithmetic-operations)
//!
//! </td><td>
//!
//! element-wise arithmetic operations
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `a**3`
//!
//! </td><td>
//!
//! [`a.mapv(|a| a.powi(3))`][.mapv()]
//!
//! </td><td>
//!
//! element-wise power of 3
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.sqrt(a)`
//!
//! </td><td>
//!
//! [`a.mapv(f64::sqrt)`][.mapv()]
//!
//! </td><td>
//!
//! element-wise square root for `f64` array
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `(a>0.5)`
//!
//! </td><td>
//!
//! [`a.mapv(|a| a > 0.5)`][.mapv()]
//!
//! </td><td>
//!
//! array of `bool`s of same shape as `a` with `true` where `a > 0.5` and `false` elsewhere
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.sum(a)` or `a.sum()`
//!
//! </td><td>
//!
//! [`a.sum()`][.sum()]
//!
//! </td><td>
//!
//! sum the elements in `a`
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.sum(a, axis=2)` or `a.sum(axis=2)`
//!
//! </td><td>
//!
//! [`a.sum_axis(Axis(2))`][.sum_axis()]
//!
//! </td><td>
//!
//! sum the elements in `a` along axis 2
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.mean(a)` or `a.mean()`
//!
//! </td><td>
//!
//! `a.sum() / a.len() as f64`
//!
//! </td><td>
//!
//! calculate the mean of the elements in `f64` array `a`
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.mean(a, axis=2)` or `a.mean(axis=2)`
//!
//! </td><td>
//!
//! [`a.mean_axis(Axis(2))`][.mean_axis()]
//!
//! </td><td>
//!
//! calculate the mean of the elements in `a` along axis 2
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.allclose(a, b, atol=1e-8)`
//!
//! </td><td>
//!
//! [`a.abs_diff_eq(&b, 1e-8)`][.abs_diff_eq()]
//!
//! </td><td>
//!
//! check if the arrays' elementwise differences are within an absolute tolerance
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.diag(a)`
//!
//! </td><td>
//!
//! [`a.diag()`][.diag()]
//!
//! </td><td>
//!
//! view the diagonal of `a`
//!
//! </td></tr>
//!
//! <tr><td>
//!
//! `np.linalg`
//!
//! </td><td>
//!
//! See other crates, e.g.
//! [`ndarray-linalg`](https://crates.io/crates/ndarray-linalg) and
//! [`linxal`](https://crates.io/crates/linxal).
//!
//! </td><td>
//!
//! linear algebra (matrix inverse, solving, decompositions, etc.)
//!
//! </td></tr>
//! </table>
//!
//! ## Array manipulation
//!
//! NumPy | `ndarray` | Notes
//! ------|-----------|------
//! `a[:] = 3.` | [`a.fill(3.)`][.fill()] | set all array elements to the same scalar value
//! `a[:] = b` | [`a.assign(&b)`][.assign()] | copy the data from array `b` into array `a`
//! `np.concatenate((a,b), axis=1)` | [`stack![Axis(1), a, b]`][stack!] or [`stack(Axis(1), &[a.view(), b.view()])`][stack()] | concatenate arrays `a` and `b` along axis 1
//! `a[:,np.newaxis]` or `np.expand_dims(a, axis=1)` | [`a.insert_axis(Axis(1))`][.insert_axis()] | create an array from `a`, inserting a new axis 1
//! `a.transpose()` or `a.T` | [`a.t()`][.t()] or [`a.reversed_axes()`][.reversed_axes()] | transpose of array `a` (view for `.t()` or by-move for `.reversed_axes()`)
//! `np.diag(a)` | [`a.diag()`][.diag()] | view the diagonal of `a`
//! `a.flatten()` | [`Array::from_iter(a.iter())`][::from_iter()] | create a 1-D array by flattening `a`
//!
//! ## Iteration
//!
//! `ndarray` has lots of interesting iterators/producers that implement the
//! [`NdProducer`][NdProducer] trait, which is a generalization of `Iterator`
//! to multiple dimensions. This makes it possible to correctly and efficiently
//! zip together slices/subviews of arrays in multiple dimensions with
//! [`Zip`][Zip] or [`azip!()`][azip!]. The purpose of this is similar to
//! [`np.nditer`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nditer.html),
//! but [`Zip`][Zip] is implemented and used somewhat differently.
//!
//! This table lists some of the iterators/producers which have a direct
//! equivalent in NumPy. For a more complete introduction to producers and
//! iterators, see [*Loops, Producers, and
//! Iterators*](../../struct.ArrayBase.html#loops-producers-and-iterators).
//! Note that there are also variants of these iterators (with a `_mut` suffix)
//! that yield `ArrayViewMut` instead of `ArrayView`.
//!
//! NumPy | `ndarray` | Notes
//! ------|-----------|------
//! `a.flat` | [`a.iter()`][.iter()] | iterator over the array elements in logical order
//! `np.ndenumerate(a)` | [`a.indexed_iter()`][.indexed_iter()] | flat iterator yielding the index along with each element reference
//! `iter(a)` | [`a.outer_iter()`][.outer_iter()] or [`a.axis_iter(Axis(0))`][.axis_iter()] | iterator over the first (outermost) axis, yielding each subview
//!
//! ## Convenience methods for 2-D arrays
//!
//! NumPy | `ndarray` | Notes
//! ------|-----------|------
//! `len(a)` or `a.shape[0]` | [`a.nrows()`][.nrows()] | get the number of rows in a 2-D array
//! `a.shape[1]` | [`a.ncols()`][.ncols()] | get the number of columns in a 2-D array
//! `a[1]` or `a[1,:]` | [`a.row(1)`][.row()] or [`a.row_mut(1)`][.row_mut()] | view (or mutable view) of row 1 in a 2-D array
//! `a[:,4]` | [`a.column(4)`][.column()] or [`a.column_mut(4)`][.column_mut()] | view (or mutable view) of column 4 in a 2-D array
//! `a.shape[0] == a.shape[1]` | [`a.is_square()`][.is_square()] | check if the array is square
//!
//! [.abs_diff_eq()]: ../../struct.ArrayBase.html#impl-AbsDiffEq<ArrayBase<S2%2C%20D>>
//! [ArcArray]: ../../type.ArcArray.html
//! [arr2()]: ../../fn.arr2.html
//! [array!]: ../../macro.array.html
//! [Array]: ../../type.Array.html
//! [Array2]: ../../type.Array2.html
//! [ArrayBase]: ../../struct.ArrayBase.html
//! [ArrayView]: ../../type.ArrayView.html
//! [ArrayViewMut]: ../../type.ArrayViewMut.html
//! [.assign()]: ../../struct.ArrayBase.html#method.assign
//! [.axis_iter()]: ../../struct.ArrayBase.html#method.axis_iter
//! [azip!]: ../../macro.azip.html
//! [.ncols()]: ../../struct.ArrayBase.html#method.ncols
//! [.column()]: ../../struct.ArrayBase.html#method.column
//! [.column_mut()]: ../../struct.ArrayBase.html#method.column_mut
//! [CowArray]: ../../type.CowArray.html
//! [::default()]: ../../struct.ArrayBase.html#method.default
//! [.diag()]: ../../struct.ArrayBase.html#method.diag
//! [.dim()]: ../../struct.ArrayBase.html#method.dim
//! [::eye()]: ../../struct.ArrayBase.html#method.eye
//! [.fill()]: ../../struct.ArrayBase.html#method.fill
//! [.fold()]: ../../struct.ArrayBase.html#method.fold
//! [.fold_axis()]: ../../struct.ArrayBase.html#method.fold_axis
//! [::from_elem()]: ../../struct.ArrayBase.html#method.from_elem
//! [::from_iter()]: ../../struct.ArrayBase.html#method.from_iter
//! [::from_shape_fn()]: ../../struct.ArrayBase.html#method.from_shape_fn
//! [::from_shape_vec()]: ../../struct.ArrayBase.html#method.from_shape_vec
//! [::from_shape_vec_unchecked()]: ../../struct.ArrayBase.html#method.from_shape_vec_unchecked
//! [::from_vec()]: ../../struct.ArrayBase.html#method.from_vec
//! [.index()]: ../../struct.ArrayBase.html#impl-Index<I>
//! [.indexed_iter()]: ../../struct.ArrayBase.html#method.indexed_iter
//! [.insert_axis()]: ../../struct.ArrayBase.html#method.insert_axis
//! [.is_empty()]: ../../struct.ArrayBase.html#method.is_empty
//! [.is_square()]: ../../struct.ArrayBase.html#method.is_square
//! [.iter()]: ../../struct.ArrayBase.html#method.iter
//! [Ix]: ../../type.Ix.html
//! [.len()]: ../../struct.ArrayBase.html#method.len
//! [.len_of()]: ../../struct.ArrayBase.html#method.len_of
//! [::linspace()]: ../../struct.ArrayBase.html#method.linspace
//! [.map()]: ../../struct.ArrayBase.html#method.map
//! [.map_axis()]: ../../struct.ArrayBase.html#method.map_axis
//! [.map_inplace()]: ../../struct.ArrayBase.html#method.map_inplace
//! [.mapv()]: ../../struct.ArrayBase.html#method.mapv
//! [.mapv_inplace()]: ../../struct.ArrayBase.html#method.mapv_inplace
//! [.mapv_into()]: ../../struct.ArrayBase.html#method.mapv_into
//! [matrix-* dot]: ../../struct.ArrayBase.html#method.dot-1
//! [.mean_axis()]: ../../struct.ArrayBase.html#method.mean_axis
//! [.ndim()]: ../../struct.ArrayBase.html#method.ndim
//! [NdProducer]: ../../trait.NdProducer.html
//! [::ones()]: ../../struct.ArrayBase.html#method.ones
//! [.outer_iter()]: ../../struct.ArrayBase.html#method.outer_iter
//! [::range()]: ../../struct.ArrayBase.html#method.range
//! [.raw_dim()]: ../../struct.ArrayBase.html#method.raw_dim
//! [.reversed_axes()]: ../../struct.ArrayBase.html#method.reversed_axes
//! [.row()]: ../../struct.ArrayBase.html#method.row
//! [.row_mut()]: ../../struct.ArrayBase.html#method.row_mut
//! [.nrows()]: ../../struct.ArrayBase.html#method.nrows
//! [s!]: ../../macro.s.html
//! [.sum()]: ../../struct.ArrayBase.html#method.sum
//! [.slice()]: ../../struct.ArrayBase.html#method.slice
//! [.slice_axis()]: ../../struct.ArrayBase.html#method.slice_axis
//! [.slice_collapse()]: ../../struct.ArrayBase.html#method.slice_collapse
//! [.slice_move()]: ../../struct.ArrayBase.html#method.slice_move
//! [.slice_mut()]: ../../struct.ArrayBase.html#method.slice_mut
//! [.shape()]: ../../struct.ArrayBase.html#method.shape
//! [stack!]: ../../macro.stack.html
//! [stack()]: ../../fn.stack.html
//! [.strides()]: ../../struct.ArrayBase.html#method.strides
//! [.index_axis()]: ../../struct.ArrayBase.html#method.index_axis
//! [.sum_axis()]: ../../struct.ArrayBase.html#method.sum_axis
//! [.t()]: ../../struct.ArrayBase.html#method.t
//! [::uninitialized()]: ../../struct.ArrayBase.html#method.uninitialized
//! [vec-* dot]: ../../struct.ArrayBase.html#method.dot
//! [.visit()]: ../../struct.ArrayBase.html#method.visit
//! [::zeros()]: ../../struct.ArrayBase.html#method.zeros
//! [Zip]: ../../struct.Zip.html

pub mod coord_transform;
pub mod rk_step;
pub mod simple_math;
