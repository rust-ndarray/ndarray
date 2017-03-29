// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![crate_name="ndarray"]
#![doc(html_root_url = "http://bluss.github.io/rust-ndarray/master/")]

//! The `ndarray` crate provides an N-dimensional container for general elements
//! and for numerics.
//!
//! - [**`ArrayBase`**](struct.ArrayBase.html):
//!   The N-dimensional array type itself.<br>
//!   It is used to implement both the owned arrays and the views; see its docs
//!   for an overview of all array features.  
//! - The main specific array type is [**`Array`**](type.Array.html), which owns
//! its elements.
//!
//! ## Highlights
//!
//! - Generic N-dimensional array
//! - Slicing, also with arbitrary step size, and negative indices to mean
//!   elements from the end of the axis.
//! - Views and subviews of arrays; iterators that yield subviews.
//! - Higher order operations and arithmetic are performant
//! - Array views can be used to slice and mutate any `[T]` data using
//!   `ArrayView::from` and `ArrayViewMut::from`.
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
//!
//! ## Crate Feature Flags
//!
//! The following crate feature flags are available. They are configured in your
//! `Cargo.toml`.
//!
//! - `rustc-serialize`
//!   - Optional, compatible with Rust stable
//!   - Enables serialization support for rustc-serialize 0.3
//! - `serde`
//!   - Optional, compatible with Rust stable
//!   - Enables serialization support for serde 0.9
//! - `blas`
//!   - Optional and experimental, compatible with Rust stable
//!   - Enable transparent BLAS support for matrix multiplication.
//!     Uses ``blas-sys`` for pluggable backend, which needs to be configured
//!     separately.
//!

#[cfg(feature = "serde")]
extern crate serde;
#[cfg(feature = "rustc-serialize")]
extern crate rustc_serialize as serialize;

#[cfg(feature="blas")]
extern crate blas_sys;

extern crate matrixmultiply;

extern crate itertools;
extern crate num_traits as libnum;
extern crate num_complex;

use std::iter::Zip as ZipIter;
use std::marker::PhantomData;
use std::rc::Rc;
use std::slice::{self, Iter as SliceIter, IterMut as SliceIterMut};

pub use dimension::{
    Dimension,
    IntoDimension,
    RemoveAxis,
    Axis,
    Axes,
    AxisDescription,
};
pub use dimension::dim::*;

pub use dimension::NdIndex;
pub use indexes::Indices;
pub use indexes::{indices, indices_of};
pub use error::{ShapeError, ErrorKind};
pub use si::{Si, S};

use iterators::Baseiter;
pub use iterators::{
    InnerIter,
    InnerIterMut,
    AxisIter,
    AxisIterMut,
    AxisChunksIter,
    AxisChunksIterMut,
    WholeChunks,
    WholeChunksIter,
};

pub use arraytraits::AsArray;
pub use linalg_traits::{LinalgScalar, NdFloat};
pub use stacking::stack;

pub use shape_builder::{ ShapeBuilder};

#[macro_use] mod macro_utils;
#[macro_use] mod private;
mod aliases;
mod arraytraits;
#[cfg(feature = "serde")]
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

mod indexes;
mod iterators;
mod linalg_traits;
mod linspace;
mod numeric_util;
mod si;
mod error;
mod shape_builder;
mod stacking;
mod zip;

pub use zip::{
    Zip,
    NdProducer,
    IntoNdProducer,
    Layout,
    FoldWhile,
};

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
    };
    pub use dimension::DimensionExt;
    /// Wrapper type for private methods
    #[derive(Copy, Clone, Debug)]
    pub struct Priv<T>(pub T);
}

pub mod prelude;

/// Array index type
pub type Ix = usize;
/// Array index type (signed)
pub type Ixs = isize;

/// An *N*-dimensional array.
///
/// The array is a general container of elements. It cannot grow or shrink, but
/// can be sliced into subsets of its data.
/// The array supports arithmetic operations by applying them elementwise.
///
/// The `ArrayBase<S, D>` is parameterized by `S` for the data container and
/// `D` for the dimensionality.
///
/// Type aliases [`Array`], [`RcArray`], [`ArrayView`], and [`ArrayViewMut`] refer
/// to `ArrayBase` with different types for the data container.
///
/// [`Array`]: type.Array.html
/// [`RcArray`]: type.RcArray.html
/// [`ArrayView`]: type.ArrayView.html
/// [`ArrayViewMut`]: type.ArrayViewMut.html
///
/// ## Contents
///
/// + [Array](#array)
/// + [RcArray](#rcarray)
/// + [Array Views](#array-views)
/// + [Indexing and Dimension](#indexing-and-dimension)
/// + [Slicing](#slicing)
/// + [Subviews](#subviews)
/// + [Arithmetic Operations](#arithmetic-operations)
/// + [Broadcasting](#broadcasting)
/// + [Constructor Methods for Owned Arrays](#constructor-methods-for-owned-arrays)
/// + [Methods For All Array Types](#methods-for-all-array-types)
/// + [Methods Specific to Array Views](#methods-specific-to-array-views)
///
///
///
///
/// ## `Array`
///
/// [`Array`](type.Array.html) is an owned array that ows the underlying array
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
/// ## `RcArray`
///
/// [`RcArray`](type.RcArray.html) is an owned array with reference counted
/// data (shared ownership).
/// Sharing requires that it uses copy-on-write for mutable operations.
/// Calling a method for mutating elements on `RcArray`, for example
/// [`view_mut()`](#method.view_mut) or [`get_mut()`](#method.get_mut),
/// will break sharing and require a clone of the data (if it is not uniquely held).
///
/// ## Array Views
///
/// `ArrayView` and `ArrayViewMut` are read-only and read-write array views
/// respectively. They use dimensionality, indexing, and almost all other
/// methods the same was as the other array types.
///
/// A view is created from an array using `.view()`, `.view_mut()`, using
/// slicing (`.slice()`, `.slice_mut()`) or from one of the many iterators
/// that yield array views.
///
/// You can also create an array view from a regular slice of data not
/// allocated with `Array` — see [Methods Specific to Array
/// Views](#methods-specific-to-array-views).
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
/// ## Slicing
///
/// You can use slicing to create a view of a subset of the data in
/// the array. Slicing methods include `.slice()`, `.islice()`,
/// `.slice_mut()`.
///
/// The slicing argument can be passed using the macro [`s![]`](macro.s!.html),
/// which will be used in all examples. (The explicit form is a reference
/// to a fixed size array of [`Si`]; see its docs for more information.)
/// [`Si`]: struct.Si.html
///
/// ```
/// // import the s![] macro
/// #[macro_use(s)]
/// extern crate ndarray;
///
/// use ndarray::arr3;
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
/// // without the macro, the explicit argument is `&[S, Si(0, Some(1), 1), S]`
///
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
/// }
/// ```
///
/// ## Subviews
///
/// Subview methods allow you to restrict the array view while removing
/// one axis from the array. Subview methods include `.subview()`,
/// `.isubview()`, `.subview_mut()`.
///
/// Subview takes two arguments: `axis` and `index`.
///
/// ```
/// use ndarray::{arr3, aview2, Axis};
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
/// ```
///
/// `.isubview()` modifies the view in the same way as `subview()`, but
/// since it is *in place*, it cannot remove the collapsed axis. It becomes
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
/// Let `A` be an array or view of any kind. Let `B` be an array
/// with owned storage (either `Array` or `RcArray`).
/// Let `C` be an array with mutable data (either `Array`, `RcArray`
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
/// The trait [`ScalarOperand`](trait.ScalarOperand.html) marks types that can be used in arithmetic
/// with arrays directly. For a scalar `K` the following combinations of operands
/// are supported (scalar can be on either the left or right side, but
/// `ScalarOperand` docs has the detailed condtions).
///
/// - `&A @ K` or `K @ &A` which produces a new `Array`
/// - `B @ K` or `K @ B` which consumes `B`, updates it with the result and returns it
/// - `C @= K` which performs an arithmetic operation in place
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
/// It can act as both an owner as the data as well as a shared reference (view
/// like).
pub type RcArray<A, D> = ArrayBase<Rc<Vec<A>>, D>;

/// An array that owns its data uniquely.
///
/// `Array` is the main n-dimensional array type, and it owns all its array
/// elements.
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
/// [`Array3`](Array3.t.html) and so on.
pub type Array<A, D> = ArrayBase<Vec<A>, D>;

/// A lightweight array view.
///
/// An array view represents an array or a part of it, created from
/// an iterator, subview or slice of an array.
///
/// Array views have all the methods of an array (see [`ArrayBase`][ab]).
///
/// See also [**Methods Specific To Array Views**](struct.ArrayBase.html#methods-specific-to-array-views)
///
/// [ab]: struct.ArrayBase.html
pub type ArrayView<'a, A, D> = ArrayBase<ViewRepr<&'a A>, D>;
/// A lightweight read-write array view.
///
/// An array view represents an array or a part of it, created from
/// an iterator, subview or slice of an array.
///
/// Array views have all the methods of an array (see [`ArrayBase`][ab]).
///
/// See also [**Methods Specific To Array Views**](struct.ArrayBase.html#methods-specific-to-array-views)
///
/// [ab]: struct.ArrayBase.html
pub type ArrayViewMut<'a, A, D> = ArrayBase<ViewRepr<&'a mut A>, D>;

/// Array view’s representation.
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
        for row in self.inner_iter_mut() {
            row.into_iter_().fold((), |(), elt| f(elt));
        }
    }

    /// Remove array axis `axis` and return the result.
    fn try_remove_axis(self, axis: Axis) -> ArrayBase<S, D::TrySmaller>
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

}


mod impl_1d;
mod impl_2d;

mod numeric;

pub mod linalg;

mod impl_ops;
pub use impl_ops::ScalarOperand;

// Array view methods
mod impl_views;

/// Private array view methods
impl<'a, A, D> ArrayBase<ViewRepr<&'a A>, D>
    where D: Dimension,
{
    /// Create a new `ArrayView`
    ///
    /// Unsafe because: `ptr` must be valid for the given dimension and strides.
    #[inline(always)]
    unsafe fn new_(ptr: *const A, dim: D, strides: D) -> Self {
        ArrayView {
            data: ViewRepr::new(),
            ptr: ptr as *mut A,
            dim: dim,
            strides: strides,
        }
    }

    #[inline]
    fn into_base_iter(self) -> Baseiter<'a, A, D> {
        unsafe {
            Baseiter::new(self.ptr, self.dim, self.strides)
        }
    }

    #[inline]
    fn into_elements_base(self) -> ElementsBase<'a, A, D> {
        ElementsBase { inner: self.into_base_iter() }
    }

    fn into_iter_(self) -> Iter<'a, A, D> {
        Iter {
            inner: if let Some(slc) = self.into_slice() {
                ElementsRepr::Slice(slc.iter())
            } else {
                ElementsRepr::Counted(self.into_elements_base())
            },
        }
    }

    /// Return an outer iterator for this view.
    #[doc(hidden)] // not official
    #[deprecated(note="This method will be replaced.")]
    pub fn into_outer_iter(self) -> AxisIter<'a, A, D::Smaller>
        where D: RemoveAxis,
    {
        iterators::new_outer_iter(self)
    }

}

impl<'a, A, D> ArrayBase<ViewRepr<&'a mut A>, D>
    where D: Dimension,
{
    /// Create a new `ArrayView`
    ///
    /// Unsafe because: `ptr` must be valid for the given dimension and strides.
    #[inline(always)]
    unsafe fn new_(ptr: *mut A, dim: D, strides: D) -> Self {
        ArrayViewMut {
            data: ViewRepr::new(),
            ptr: ptr,
            dim: dim,
            strides: strides,
        }
    }

    #[inline]
    fn into_base_iter(self) -> Baseiter<'a, A, D> {
        unsafe {
            Baseiter::new(self.ptr, self.dim, self.strides)
        }
    }

    #[inline]
    fn into_elements_base(self) -> ElementsBaseMut<'a, A, D> {
        ElementsBaseMut { inner: self.into_base_iter() }
    }

    fn into_iter_(self) -> IterMut<'a, A, D> {
        IterMut {
            inner:
                if self.is_standard_layout() {
                    let slc = unsafe {
                        slice::from_raw_parts_mut(self.ptr, self.len())
                    };
                    ElementsRepr::Slice(slc.iter_mut())
                } else {
                    ElementsRepr::Counted(self.into_elements_base())
                }
        }
    }

    /// Return an outer iterator for this view.
    #[doc(hidden)] // not official
    #[deprecated(note="This method will be replaced.")]
    pub fn into_outer_iter(self) -> AxisIterMut<'a, A, D::Smaller>
        where D: RemoveAxis,
    {
        iterators::new_outer_iter_mut(self)
    }
}


/// An iterator over the elements of an array.
///
/// Iterator element type is `&'a A`.
///
/// See [`.iter()`](struct.ArrayBase.html#method.iter) for more information.
pub struct Iter<'a, A: 'a, D> {
    inner: ElementsRepr<SliceIter<'a, A>, ElementsBase<'a, A, D>>,
}

/// Counted read only iterator
struct ElementsBase<'a, A: 'a, D> {
    inner: Baseiter<'a, A, D>,
}

/// An iterator over the elements of an array (mutable).
///
/// Iterator element type is `&'a mut A`.
///
/// See [`.iter_mut()`](struct.ArrayBase.html#method.iter_mut) for more information.
pub struct IterMut<'a, A: 'a, D> {
    inner: ElementsRepr<SliceIterMut<'a, A>, ElementsBaseMut<'a, A, D>>,
}

/// An iterator over the elements of an array.
///
/// Iterator element type is `&'a mut A`.
struct ElementsBaseMut<'a, A: 'a, D> {
    inner: Baseiter<'a, A, D>,
}

/// An iterator over the indexes and elements of an array.
///
/// See [`.indexed_iter()`](struct.ArrayBase.html#method.indexed_iter) for more information.
#[derive(Clone)]
pub struct IndexedIter<'a, A: 'a, D>(ElementsBase<'a, A, D>);
/// An iterator over the indexes and elements of an array (mutable).
///
/// See [`.indexed_iter_mut()`](struct.ArrayBase.html#method.indexed_iter_mut) for more information.
pub struct IndexedIterMut<'a, A: 'a, D>(ElementsBaseMut<'a, A, D>);

fn zipsl<'a, 'b, A, B>(t: &'a [A], u: &'b [B])
    -> ZipIter<SliceIter<'a, A>, SliceIter<'b, B>> {
    t.iter().zip(u)
}
fn zipsl_mut<'a, 'b, A, B>(t: &'a mut [A], u: &'b mut [B])
    -> ZipIter<SliceIterMut<'a, A>, SliceIterMut<'b, B>> {
    t.iter_mut().zip(u)
}

use itertools::{cons_tuples, ConsTuples};

trait ZipExt : Iterator {
    fn zip_cons<J>(self, iter: J) -> ConsTuples<ZipIter<Self, J::IntoIter>, (Self::Item, J::Item)>
        where J: IntoIterator,
              Self: Sized,
    {
        cons_tuples(self.zip(iter))
    }
}

impl<I> ZipExt for I where I: Iterator { }

enum ElementsRepr<S, C> {
    Slice(S),
    Counted(C),
}


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
