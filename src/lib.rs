// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![crate_name="ndarray"]
#![doc(html_root_url = "https://docs.rs/ndarray/0.11/")]

//! The `ndarray` crate provides an *n*-dimensional container for general elements
//! and for numerics.
//!
//! In *n*-dimensional we include for example 1-dimensional rows or columns,
//! 2-dimensional matrices, and higher dimensional arrays. If the array has *n*
//! dimensions, then an element in the array is accessed by using that many indices.
//! Each dimension is also called an *axis*.
//!
//! - **[`ArrayBase`](struct.ArrayBase.html)**:
//!   The *n*-dimensional array type itself.<br>
//!   It is used to implement both the owned arrays and the views; see its docs
//!   for an overview of all array features.<br>
//! - The main specific array type is **[`Array`](type.Array.html)**, which owns
//! its elements.
//!
//! ## Highlights
//!
//! - Generic *n*-dimensional array
//! - Slicing, also with arbitrary step size, and negative indices to mean
//!   elements from the end of the axis.
//! - Views and subviews of arrays; iterators that yield subviews.
//! - Higher order operations and arithmetic are performant
//! - Array views can be used to slice and mutate any `[T]` data using
//!   `ArrayView::from` and `ArrayViewMut::from`.
//! - `Zip` for lock step function application across two or more arrays or other
//!   item producers (`NdProducer` trait).
//!
//! ## Crate Status
//!
//! - Still iterating on and evolving the crate
//!   + The crate is continuously developing, and breaking changes are expected
//!     during evolution from version to version. We adopt the newest stable
//!     rust features if we need them.
//! - Performance:
//!   + Prefer higher order methods and arithmetic operations on arrays first,
//!     then iteration, and as a last priority using indexed algorithms.
//!   + The higher order functions like ``.map()``, ``.map_inplace()``,
//!     ``.zip_mut_with()``, ``Zip`` and ``azip!()`` are the most efficient ways
//!     to perform single traversal and lock step traversal respectively.
//!   + Performance of an operation depends on the memory layout of the array
//!     or array view. Especially if it's a binary operation, which
//!     needs matching memory layout to be efficient (with some exceptions).
//!   + Efficient floating point matrix multiplication even for very large
//!     matrices; can optionally use BLAS to improve it further.
//!   + See also the [`ndarray-parallel`] crate for integration with rayon.
//! - **Requires Rust 1.22**
//!
//! [`ndarray-parallel`]: https://docs.rs/ndarray-parallel
//!
//! ## Crate Feature Flags
//!
//! The following crate feature flags are available. They are configured in your
//! `Cargo.toml`.
//!
//! - `rustc-serialize`
//!   - Optional, compatible with Rust stable
//!   - Enables serialization support for rustc-serialize 0.3
//! - `serde-1`
//!   - Optional, compatible with Rust stable
//!   - Enables serialization support for serde 1.0
//! - `blas`
//!   - Optional and experimental, compatible with Rust stable
//!   - Enable transparent BLAS support for matrix multiplication.
//!     Uses ``blas-src`` for pluggable backend, which needs to be configured
//!     separately.
//!
//! ## Documentation
//!
//! * The docs for [`ArrayBase`](struct.ArrayBase.html) provide an overview of
//!   the *n*-dimensional array type. Other good pages to look at are the
//!   documentation for the [`s![]`](macro.s.html) and
//!   [`azip!()`](macro.azip.html) macros.
//!
//! * If you have experience with NumPy, you may also be interested in
//!   [`ndarray_for_numpy_users`](doc/ndarray_for_numpy_users/index.html).

#[cfg(feature = "serde-1")]
extern crate serde;
#[cfg(feature = "rustc-serialize")]
extern crate rustc_serialize as serialize;

#[cfg(feature="blas")]
extern crate cblas_sys;
#[cfg(feature="blas")]
extern crate blas_src;

extern crate matrixmultiply;

#[macro_use(izip)] extern crate itertools;
extern crate num_traits as libnum;
extern crate num_complex;

#[cfg(feature = "docs")]
pub mod doc;

use std::marker::PhantomData;
use std::sync::Arc;

pub use dimension::{
    Dimension,
    IntoDimension,
    RemoveAxis,
    Axis,
    AxisDescription,
};
pub use dimension::dim::*;

pub use dimension::NdIndex;
pub use dimension::IxDynImpl;
pub use indexes::{indices, indices_of};
pub use error::{ShapeError, ErrorKind};
pub use slice::{Slice, SliceInfo, SliceNextDim, SliceOrIndex};

use iterators::Baseiter;
use iterators::{ElementsBase, ElementsBaseMut, Iter, IterMut};

pub use arraytraits::AsArray;
pub use linalg_traits::{LinalgScalar, NdFloat};
pub use stacking::stack;

pub use shape_builder::{ ShapeBuilder};
pub use impl_views::IndexLonger;

#[macro_use] mod macro_utils;
#[macro_use] mod private;
mod aliases;
mod arraytraits;
#[cfg(feature = "serde-1")]
mod array_serde;
#[cfg(feature = "rustc-serialize")]
mod array_serialize;
mod arrayformat;
mod data_traits;

pub use aliases::*;

pub use data_traits::{
    Data,
    DataMut,
    DataOwned,
    DataShared,
    DataClone,
};

mod dimension;

mod free_functions;
pub use free_functions::*;
pub use iterators::iter;

#[macro_use]
mod slice;
mod layout;
mod indexes;
mod iterators;
mod linalg_traits;
mod linspace;
mod numeric_util;
mod error;
mod shape_builder;
mod stacking;
mod zip;

pub use zip::{
    Zip,
    NdProducer,
    IntoNdProducer,
    FoldWhile,
};

pub use layout::Layout;

/// Implementation's prelude. Common types used everywhere.
mod imp_prelude {
    pub use prelude::*;
    pub use {
        RemoveAxis,
        Data,
        DataMut,
        DataOwned,
        DataShared,
        ViewRepr,
        Ix, Ixs,
    };
    pub use dimension::DimensionExt;
}

pub mod prelude;

/// Array index type
pub type Ix = usize;
/// Array index type (signed)
pub type Ixs = isize;

/// An *n*-dimensional array.
///
/// The array is a general container of elements. It cannot grow or shrink, but
/// can be sliced into subsets of its data.
/// The array supports arithmetic operations by applying them elementwise.
///
/// In *n*-dimensional we include for example 1-dimensional rows or columns,
/// 2-dimensional matrices, and higher dimensional arrays. If the array has *n*
/// dimensions, then an element is accessed by using that many indices.
///
/// The `ArrayBase<S, D>` is parameterized by `S` for the data container and
/// `D` for the dimensionality.
///
/// Type aliases [`Array`], [`ArcArray`], [`ArrayView`], and [`ArrayViewMut`] refer
/// to `ArrayBase` with different types for the data container.
///
/// [`Array`]: type.Array.html
/// [`ArcArray`]: type.ArcArray.html
/// [`ArrayView`]: type.ArrayView.html
/// [`ArrayViewMut`]: type.ArrayViewMut.html
///
/// ## Contents
///
/// + [Array](#array)
/// + [ArcArray](#arcarray)
/// + [Array Views](#array-views)
/// + [Indexing and Dimension](#indexing-and-dimension)
/// + [Loops, Producers and Iterators](#loops-producers-and-iterators)
/// + [Slicing](#slicing)
/// + [Subviews](#subviews)
/// + [Arithmetic Operations](#arithmetic-operations)
/// + [Broadcasting](#broadcasting)
/// + [Constructor Methods for Owned Arrays](#constructor-methods-for-owned-arrays)
/// + [Methods For All Array Types](#methods-for-all-array-types)
///
///
/// ## `Array`
///
/// [`Array`](type.Array.html) is an owned array that owns the underlying array
/// elements directly (just like a `Vec`) and it is the default way to create and
/// store n-dimensional data. `Array<A, D>` has two type parameters: `A` for
/// the element type, and `D` for the dimensionality. A particular
/// dimensionality's type alias like `Array3<A>` just has the type parameter
/// `A` for element type.
///
/// An example:
///
/// ```
/// // Create a three-dimensional f64 array, initialized with zeros
/// use ndarray::Array3;
/// let mut temperature = Array3::<f64>::zeros((3, 4, 5));
/// // Increase the temperature in this location
/// temperature[[2, 2, 2]] += 0.5;
/// ```
///
/// ## `ArcArray`
///
/// [`ArcArray`](type.ArcArray.html) is an owned array with reference counted
/// data (shared ownership).
/// Sharing requires that it uses copy-on-write for mutable operations.
/// Calling a method for mutating elements on `ArcArray`, for example
/// [`view_mut()`](#method.view_mut) or [`get_mut()`](#method.get_mut),
/// will break sharing and require a clone of the data (if it is not uniquely held).
///
/// ## Array Views
///
/// [`ArrayView`] and [`ArrayViewMut`] are read-only and read-write array views
/// respectively. They use dimensionality, indexing, and almost all other
/// methods the same was as the other array types.
///
/// Methods for `ArrayBase` apply to array views too, when the trait bounds
/// allow.
///
/// Please see the documentation for the respective array view for an overview
/// of methods specific to array views: [`ArrayView`], [`ArrayViewMut`].
///
/// A view is created from an array using `.view()`, `.view_mut()`, using
/// slicing (`.slice()`, `.slice_mut()`) or from one of the many iterators
/// that yield array views.
///
/// You can also create an array view from a regular slice of data not
/// allocated with `Array` — see array view methods or their `From` impls.
///
/// Note that all `ArrayBase` variants can change their view (slicing) of the
/// data freely, even when their data can’t be mutated.
///
/// ## Indexing and Dimension
///
/// The dimensionality of the array determines the number of *axes*, for example
/// a 2D array has two axes. These are listed in “big endian” order, so that
/// the greatest dimension is listed first, the lowest dimension with the most
/// rapidly varying index is the last.
///
/// In a 2D array the index of each element is `[row, column]` as seen in this
/// 4 × 3 example:
///
/// ```ignore
/// [[ [0, 0], [0, 1], [0, 2] ],  // row 0
///  [ [1, 0], [1, 1], [1, 2] ],  // row 1
///  [ [2, 0], [2, 1], [2, 2] ],  // row 2
///  [ [3, 0], [3, 1], [3, 2] ]]  // row 3
/// //    \       \       \
/// //   column 0  \     column 2
/// //            column 1
/// ```
///
/// The number of axes for an array is fixed by its `D` type parameter: `Ix1`
/// for a 1D array, `Ix2` for a 2D array etc. The dimension type `IxDyn` allows
/// a dynamic number of axes.
///
/// A fixed size array (`[usize; N]`) of the corresponding dimensionality is
/// used to index the `Array`, making the syntax `array[[` i, j,  ...`]]`
///
/// ```
/// use ndarray::Array2;
/// let mut array = Array2::zeros((4, 3));
/// array[[1, 1]] = 7;
/// ```
///
/// Important traits and types for dimension and indexing:
///
/// - A [`Dim`](Dim.t.html) value represents a dimensionality or index.
/// - Trait [`Dimension`](Dimension.t.html) is implemented by all
/// dimensionalities. It defines many operations for dimensions and indices.
/// - Trait [`IntoDimension`](IntoDimension.t.html) is used to convert into a
/// `Dim` value.
/// - Trait [`ShapeBuilder`](ShapeBuilder.t.html) is an extension of
/// `IntoDimension` and is used when constructing an array. A shape describes
/// not just the extent of each axis but also their strides.
/// - Trait [`NdIndex`](NdIndex.t.html) is an extension of `Dimension` and is
/// for values that can be used with indexing syntax.
///
///
/// The default memory order of an array is *row major* order (a.k.a “c” order),
/// where each row is contiguous in memory.
/// A *column major* (a.k.a. “f” or fortran) memory order array has
/// columns (or, in general, the outermost axis) with contiguous elements.
///
/// The logical order of any array’s elements is the row major order
/// (the rightmost index is varying the fastest).
/// The iterators `.iter(), .iter_mut()` always adhere to this order, for example.
///
/// ## Loops, Producers and Iterators
///
/// Using [`Zip`](struct.Zip.html) is the most general way to apply a procedure
/// across one or several arrays or *producers*.
///
/// [`NdProducer`](trait.NdProducer.html) is like an iterable but for
/// multidimensional data. All producers have dimensions and axes, like an
/// array view, and they can be split and used with parallelization using `Zip`.
///
/// For example, `ArrayView<A, D>` is a producer, it has the same dimensions
/// as the array view and for each iteration it produces a reference to
/// the array element (`&A` in this case).
///
/// Another example, if we have a 10 × 10 array and use `.exact_chunks((2, 2))`
/// we get a producer of chunks which has the dimensions 5 × 5 (because
/// there are *10 / 2 = 5* chunks in either direction). The 5 × 5 chunks producer
/// can be paired with any other producers of the same dimension with `Zip`, for
/// example 5 × 5 arrays.
///
/// ### `.iter()` and `.iter_mut()`
///
/// These are the element iterators of arrays and they produce an element
/// sequence in the logical order of the array, that means that the elements
/// will be visited in the sequence that corresponds to increasing the
/// last index first: *0, ..., 0,  0*; *0, ..., 0, 1*; *0, ...0, 2* and so on.
///
/// ### `.outer_iter()` and `.axis_iter()`
///
/// These iterators produce array views of one smaller dimension.
///
/// For example, for a 2D array, `.outer_iter()` will produce the 1D rows.
/// For a 3D array, `.outer_iter()` produces 2D subviews.
///
/// `.axis_iter()` is like `outer_iter()` but allows you to pick which
/// axis to traverse.
///
/// The `outer_iter` and `axis_iter` are one dimensional producers.
///
/// ## `.genrows()`, `.gencolumns()` and `.lanes()`
///
/// [`.genrows()`][gr] is a producer (and iterable) of all rows in an array.
///
/// ```
/// use ndarray::Array;
///
/// // 1. Loop over the rows of a 2D array
/// let mut a = Array::zeros((10, 10));
/// for mut row in a.genrows_mut() {
///     row.fill(1.);
/// }
///
/// // 2. Use Zip to pair each row in 2D `a` with elements in 1D `b`
/// use ndarray::Zip;
/// let mut b = Array::zeros(a.rows());
///
/// Zip::from(a.genrows())
///     .and(&mut b)
///     .apply(|a_row, b_elt| {
///         *b_elt = a_row[a.cols() - 1] - a_row[0];
///     });
/// ```
///
/// The *lanes* of an array are 1D segments along an axis and when pointed
/// along the last axis they are *rows*, when pointed along the first axis
/// they are *columns*.
///
/// A *m* × *n* array has *m* rows each of length *n* and conversely
/// *n* columns each of length *m*.
///
/// To generalize this, we say that an array of dimension *a* × *m* × *n*
/// has *a m* rows. It's composed of *a* times the previous array, so it
/// has *a* times as many rows.
///
/// All methods: [`.genrows()`][gr], [`.genrows_mut()`][grm],
/// [`.gencolumns()`][gc], [`.gencolumns_mut()`][gcm],
/// [`.lanes(axis)`][l], [`.lanes_mut(axis)`][lm].
///
/// [gr]: #method.genrows
/// [grm]: #method.genrows_mut
/// [gc]: #method.gencolumns
/// [gcm]: #method.gencolumns_mut
/// [l]: #method.lanes
/// [lm]: #method.lanes_mut
///
/// Yes, for 2D arrays `.genrows()` and `.outer_iter()` have about the same
/// effect:
///
///  + `genrows()` is a producer with *n* - 1 dimensions of 1 dimensional items
///  + `outer_iter()` is a producer with 1 dimension of *n* - 1 dimensional items
///
/// ## Slicing
///
/// You can use slicing to create a view of a subset of the data in
/// the array. Slicing methods include [`.slice()`], [`.slice_mut()`],
/// [`.slice_move()`], and [`.slice_inplace()`].
///
/// The slicing argument can be passed using the macro [`s![]`](macro.s!.html),
/// which will be used in all examples. (The explicit form is an instance of
/// [`&SliceInfo`]; see its docs for more information.)
///
/// [`&SliceInfo`]: struct.SliceInfo.html
///
/// If a range is used, the axis is preserved. If an index is used, a subview
/// is taken with respect to the axis. See [*Subviews*](#subviews) for more
/// information about subviews. Note that [`.slice_inplace()`] behaves like
/// [`.subview_inplace()`] by preserving the number of dimensions.
///
/// [`.slice()`]: #method.slice
/// [`.slice_mut()`]: #method.slice_mut
/// [`.slice_move()`]: #method.slice_move
/// [`.slice_inplace()`]: #method.slice_inplace
///
/// ```
/// // import the s![] macro
/// #[macro_use(s)]
/// extern crate ndarray;
///
/// use ndarray::{arr2, arr3};
///
/// fn main() {
///
/// // 2 submatrices of 2 rows with 3 elements per row, means a shape of `[2, 2, 3]`.
///
/// let a = arr3(&[[[ 1,  2,  3],     // -- 2 rows  \_
///                 [ 4,  5,  6]],    // --         /
///                [[ 7,  8,  9],     //            \_ 2 submatrices
///                 [10, 11, 12]]]);  //            /
/// //  3 columns ..../.../.../
///
/// assert_eq!(a.shape(), &[2, 2, 3]);
///
/// // Let’s create a slice with
/// //
/// // - Both of the submatrices of the greatest dimension: `..`
/// // - Only the first row in each submatrix: `0..1`
/// // - Every element in each row: `..`
///
/// let b = a.slice(s![.., 0..1, ..]);
/// let c = arr3(&[[[ 1,  2,  3]],
///                [[ 7,  8,  9]]]);
/// assert_eq!(b, c);
/// assert_eq!(b.shape(), &[2, 1, 3]);
///
/// // Let’s create a slice with
/// //
/// // - Both submatrices of the greatest dimension: `..`
/// // - The last row in each submatrix: `-1..`
/// // - Row elements in reverse order: `..;-1`
/// let d = a.slice(s![.., -1.., ..;-1]);
/// let e = arr3(&[[[ 6,  5,  4]],
///                [[12, 11, 10]]]);
/// assert_eq!(d, e);
/// assert_eq!(d.shape(), &[2, 1, 3]);
///
/// // Let’s create a slice while taking a subview with
/// //
/// // - Both submatrices of the greatest dimension: `..`
/// // - The last row in each submatrix, removing that axis: `-1`
/// // - Row elements in reverse order: `..;-1`
/// let f = a.slice(s![.., -1, ..;-1]);
/// let g = arr2(&[[ 6,  5,  4],
///                [12, 11, 10]]);
/// assert_eq!(f, g);
/// assert_eq!(f.shape(), &[2, 3]);
/// }
/// ```
///
/// ## Subviews
///
/// Subview methods allow you to restrict the array view while removing one
/// axis from the array. Subview methods include [`.subview()`],
/// [`.subview_mut()`], [`.into_subview()`], and [`.subview_inplace()`]. You
/// can also take a subview by using a single index instead of a range when
/// slicing.
///
/// Subview takes two arguments: `axis` and `index`.
///
/// [`.subview()`]: #method.subview
/// [`.subview_mut()`]: #method.subview_mut
/// [`.into_subview()`]: #method.into_subview
/// [`.subview_inplace()`]: #method.subview_inplace
///
/// ```
/// #[macro_use(s)] extern crate ndarray;
///
/// use ndarray::{arr3, aview1, aview2, Axis};
///
/// # fn main() {
///
/// // 2 submatrices of 2 rows with 3 elements per row, means a shape of `[2, 2, 3]`.
///
/// let a = arr3(&[[[ 1,  2,  3],    // \ axis 0, submatrix 0
///                 [ 4,  5,  6]],   // /
///                [[ 7,  8,  9],    // \ axis 0, submatrix 1
///                 [10, 11, 12]]]); // /
///         //        \
///         //         axis 2, column 0
///
/// assert_eq!(a.shape(), &[2, 2, 3]);
///
/// // Let’s take a subview along the greatest dimension (axis 0),
/// // taking submatrix 0, then submatrix 1
///
/// let sub_0 = a.subview(Axis(0), 0);
/// let sub_1 = a.subview(Axis(0), 1);
///
/// assert_eq!(sub_0, aview2(&[[ 1,  2,  3],
///                            [ 4,  5,  6]]));
/// assert_eq!(sub_1, aview2(&[[ 7,  8,  9],
///                            [10, 11, 12]]));
/// assert_eq!(sub_0.shape(), &[2, 3]);
///
/// // This is the subview picking only axis 2, column 0
/// let sub_col = a.subview(Axis(2), 0);
///
/// assert_eq!(sub_col, aview2(&[[ 1,  4],
///                              [ 7, 10]]));
///
/// // You can take multiple subviews at once (and slice at the same time)
/// let double_sub = a.slice(s![1, .., 0]);
/// assert_eq!(double_sub, aview1(&[7, 10]));
/// # }
/// ```
///
/// [`.subview_inplace()`] modifies the view in the same way as [`.subview()`],
/// but since it is *in place*, it cannot remove the collapsed axis. It becomes
/// an axis of length 1.
///
/// `.outer_iter()` is an iterator of every subview along the zeroth (outer)
/// axis, while `.axis_iter()` is an iterator of every subview along a
/// specific axis.
///
/// ## Arithmetic Operations
///
/// Arrays support all arithmetic operations the same way: they apply elementwise.
///
/// Since the trait implementations are hard to overview, here is a summary.
///
/// ### Binary Operators with Two Arrays
///
/// Let `A` be an array or view of any kind. Let `B` be an array
/// with owned storage (either `Array` or `ArcArray`).
/// Let `C` be an array with mutable data (either `Array`, `ArcArray`
/// or `ArrayViewMut`).
/// The following combinations of operands
/// are supported for an arbitrary binary operator denoted by `@` (it can be
/// `+`, `-`, `*`, `/` and so on).
///
/// - `&A @ &A` which produces a new `Array`
/// - `B @ A` which consumes `B`, updates it with the result, and returns it
/// - `B @ &A` which consumes `B`, updates it with the result, and returns it
/// - `C @= &A` which performs an arithmetic operation in place
///
/// ### Binary Operators with Array and Scalar
///
/// The trait [`ScalarOperand`](trait.ScalarOperand.html) marks types that can be used in arithmetic
/// with arrays directly. For a scalar `K` the following combinations of operands
/// are supported (scalar can be on either the left or right side, but
/// `ScalarOperand` docs has the detailed condtions).
///
/// - `&A @ K` or `K @ &A` which produces a new `Array`
/// - `B @ K` or `K @ B` which consumes `B`, updates it with the result and returns it
/// - `C @= K` which performs an arithmetic operation in place
///
/// ### Unary Operators
///
/// Let `A` be an array or view of any kind. Let `B` be an array with owned
/// storage (either `Array` or `ArcArray`). The following operands are supported
/// for an arbitrary unary operator denoted by `@` (it can be `-` or `!`).
///
/// - `@&A` which produces a new `Array`
/// - `@B` which consumes `B`, updates it with the result, and returns it
///
/// ## Broadcasting
///
/// Arrays support limited *broadcasting*, where arithmetic operations with
/// array operands of different sizes can be carried out by repeating the
/// elements of the smaller dimension array. See
/// [`.broadcast()`](#method.broadcast) for a more detailed
/// description.
///
/// ```
/// use ndarray::arr2;
///
/// let a = arr2(&[[1., 1.],
///                [1., 2.],
///                [0., 3.],
///                [0., 4.]]);
///
/// let b = arr2(&[[0., 1.]]);
///
/// let c = arr2(&[[1., 2.],
///                [1., 3.],
///                [0., 4.],
///                [0., 5.]]);
/// // We can add because the shapes are compatible even if not equal.
/// // The `b` array is shape 1 × 2 but acts like a 4 × 2 array.
/// assert!(
///     c == a + b
/// );
/// ```
///
pub struct ArrayBase<S, D>
    where S: Data
{
    /// Rc data when used as view, Uniquely held data when being mutated
    data: S,
    /// A pointer into the buffer held by data, may point anywhere
    /// in its range.
    ptr: *mut S::Elem,
    /// The size of each axis
    dim: D,
    /// The element count stride per axis. To be parsed as `isize`.
    strides: D,
}

/// An array where the data has shared ownership and is copy on write.
///
/// It can act as both an owner as the data as well as a shared reference (view like).
///
/// **Note: this type alias is obsolete.** See the equivalent [`ArcArray`] instead.
// Use soon: #[deprecated(note="RcArray is replaced by ArcArray")]
pub type RcArray<A, D> = ArrayBase<OwnedRcRepr<A>, D>;

/// An array where the data has shared ownership and is copy on write.
///
/// The `ArcArray<A, D>` is parameterized by `A` for the element type and `D` for
/// the dimensionality.
///
/// It can act as both an owner as the data as well as a shared reference (view
/// like).
/// Calling a method for mutating elements on `ArcArray`, for example
/// [`view_mut()`](struct.ArrayBase.html#method.view_mut) or
/// [`get_mut()`](struct.ArrayBase.html#method.get_mut), will break sharing and
/// require a clone of the data (if it is not uniquely held).
///
/// `ArcArray` uses atomic reference counting like `Arc`, so it is `Send` and
/// `Sync` (when allowed by the element type of the array too).
///
/// [**`ArrayBase`**](struct.ArrayBase.html) is used to implement both the owned
/// arrays and the views; see its docs for an overview of all array features.
///
/// See also:
///
/// + [Constructor Methods for Owned Arrays](struct.ArrayBase.html#constructor-methods-for-owned-arrays)
/// + [Methods For All Array Types](struct.ArrayBase.html#methods-for-all-array-types)
pub type ArcArray<A, D> = ArrayBase<OwnedArcRepr<A>, D>;

/// An array that owns its data uniquely.
///
/// `Array` is the main n-dimensional array type, and it owns all its array
/// elements.
///
/// The `Array<A, D>` is parameterized by `A` for the element type and `D` for
/// the dimensionality.
///
/// [**`ArrayBase`**](struct.ArrayBase.html) is used to implement both the owned
/// arrays and the views; see its docs for an overview of all array features.
///
/// See also:
///
/// + [Constructor Methods for Owned Arrays](struct.ArrayBase.html#constructor-methods-for-owned-arrays)
/// + [Methods For All Array Types](struct.ArrayBase.html#methods-for-all-array-types)
/// + Dimensionality-specific type alises
/// [`Array1`](Array1.t.html),
/// [`Array2`](Array2.t.html),
/// [`Array3`](Array3.t.html), ...,
/// [`ArrayD`](ArrayD.t.html),
/// and so on.
pub type Array<A, D> = ArrayBase<OwnedRepr<A>, D>;

/// A read-only array view.
///
/// An array view represents an array or a part of it, created from
/// an iterator, subview or slice of an array.
///
/// The `ArrayView<'a, A, D>` is parameterized by `'a` for the scope of the
/// borrow, `A` for the element type and `D` for the dimensionality.
///
/// Array views have all the methods of an array (see [`ArrayBase`][ab]).
///
/// See also [`ArrayViewMut`](type.ArrayViewMut.html).
///
/// [ab]: struct.ArrayBase.html
pub type ArrayView<'a, A, D> = ArrayBase<ViewRepr<&'a A>, D>;

/// A read-write array view.
///
/// An array view represents an array or a part of it, created from
/// an iterator, subview or slice of an array.
///
/// The `ArrayViewMut<'a, A, D>` is parameterized by `'a` for the scope of the
/// borrow, `A` for the element type and `D` for the dimensionality.
///
/// Array views have all the methods of an array (see [`ArrayBase`][ab]).
///
/// See also [`ArrayView`](type.ArrayView.html).
///
/// [ab]: struct.ArrayBase.html
pub type ArrayViewMut<'a, A, D> = ArrayBase<ViewRepr<&'a mut A>, D>;

/// Array's representation.
///
/// *Don’t use this type directly—use the type alias
/// [`Array`](type.Array.html) for the array type!*
#[derive(Clone, Debug)]
pub struct OwnedRepr<A>(Vec<A>);

/// RcArray's representation.
///
/// *Don’t use this type directly—use the type alias
/// [`RcArray`](type.RcArray.html) for the array type!*
#[deprecated(note="RcArray is replaced by ArcArray")]
pub use self::OwnedArcRepr as OwnedRcRepr;

/// ArcArray's representation.
///
/// *Don’t use this type directly—use the type alias
/// [`ArcArray`](type.ArcArray.html) for the array type!*
#[derive(Debug)]
pub struct OwnedArcRepr<A>(Arc<Vec<A>>);

impl<A> Clone for OwnedArcRepr<A> {
    fn clone(&self) -> Self {
        OwnedArcRepr(self.0.clone())
    }
}


/// Array view’s representation.
///
/// *Don’t use this type directly—use the type aliases
/// [`ArrayView`](type.ArrayView.html)
/// / [`ArrayViewMut`](type.ArrayViewMut.html) for the array type!*
#[derive(Copy, Clone)]
// This is just a marker type, to carry the lifetime parameter.
pub struct ViewRepr<A> {
    life: PhantomData<A>,
}

impl<A> ViewRepr<A> {
    #[inline(always)]
    fn new() -> Self {
        ViewRepr { life: PhantomData }
    }
}

mod impl_clone;

mod impl_constructors;

mod impl_methods;
mod impl_owned_array;

/// Private Methods
impl<A, S, D> ArrayBase<S, D>
    where S: Data<Elem=A>, D: Dimension
{
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

    // Broadcast to dimension `E`, without checking that the dimensions match
    // (Checked in debug assertions).
    #[inline]
    fn broadcast_assume<E>(&self, dim: E) -> ArrayView<A, E>
        where E: Dimension,
    {
        let dim = dim.into_dimension();
        debug_assert_eq!(self.shape(), dim.slice());
        let ptr = self.ptr;
        let mut strides = dim.clone();
        strides.slice_mut().copy_from_slice(self.strides.slice());
        unsafe {
            ArrayView::new_(ptr, dim, strides)
        }
    }

    fn raw_strides(&self) -> D {
        self.strides.clone()
    }

    /// Apply closure `f` to each element in the array, in whatever
    /// order is the fastest to visit.
    fn unordered_foreach_mut<F>(&mut self, mut f: F)
        where S: DataMut,
              F: FnMut(&mut A)
    {
        if let Some(slc) = self.as_slice_memory_order_mut() {
            // FIXME: Use for loop when slice iterator is perf is restored
            for i in 0..slc.len() {
                f(&mut slc[i]);
            }
            return;
        }
        for row in self.inner_rows_mut() {
            row.into_iter_().fold((), |(), elt| f(elt));
        }
    }

    /// Remove array axis `axis` and return the result.
    fn try_remove_axis(self, axis: Axis) -> ArrayBase<S, D::Smaller>
    {
        let d = self.dim.try_remove_axis(axis);
        let s = self.strides.try_remove_axis(axis);
        ArrayBase {
            ptr: self.ptr,
            data: self.data,
            dim: d,
            strides: s,
        }
    }

    /// n-d generalization of rows, just like inner iter
    fn inner_rows(&self) -> iterators::Lanes<A, D::Smaller>
    {
        let n = self.ndim();
        iterators::new_lanes(self.view(), Axis(n.saturating_sub(1)))
    }

    /// n-d generalization of rows, just like inner iter
    fn inner_rows_mut(&mut self) -> iterators::LanesMut<A, D::Smaller>
        where S: DataMut
    {
        let n = self.ndim();
        iterators::new_lanes_mut(self.view_mut(), Axis(n.saturating_sub(1)))
    }
}


mod impl_1d;
mod impl_2d;

mod numeric;

pub mod linalg;

mod impl_ops;
pub use impl_ops::ScalarOperand;

// Array view methods
mod impl_views;

/// A contiguous array shape of n dimensions.
///
/// Either c- or f- memory ordered (*c* a.k.a *row major* is the default).
#[derive(Copy, Clone, Debug)]
pub struct Shape<D> {
    dim: D,
    is_c: bool,
}

/// An array shape of n dimensions in c-order, f-order or custom strides.
#[derive(Copy, Clone, Debug)]
pub struct StrideShape<D> {
    dim: D,
    strides: D,
    custom: bool,
}
