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
//! - [`ArrayBase`](struct.ArrayBase.html):
//!   The N-dimensional array type itself.
//! - [`OwnedArray`](type.OwnedArray.html):
//!   An array where the data is owned uniquely.
//! - [`RcArray`](type.RcArray.html):
//!   An array where the data has shared ownership and is copy on write.
//! - [`ArrayView`](type.ArrayView.html), [`ArrayViewMut`](type.ArrayViewMut.html):
//!   Lightweight array views.
//!
//! ## Highlights
//!
//! - Generic N-dimensional array
//! - Slicing, also with arbitrary step size, and negative indices to mean
//!   elements from the end of the axis.
//! - There is both a copy on write array (`RcArray`), or a regular uniquely owned array
//!   (`OwnedArray`), and both can use read-only and read-write array views.
//! - Iteration and most operations are efficient on arrays with contiguous
//!   innermost dimension.
//! - Array views can be used to slice and mutate any `[T]` data using
//!   `ArrayView::from` and `ArrayViewMut::from`.
//!
//! ## Crate Status
//!
//! - Still iterating on and evolving the API
//!   + The crate is continuously developing, and breaking changes are expected
//!     during evolution from version to version. We adhere to semver,
//!     but alpha releases break at will.
//!   + We adopt the newest stable rust features we need.
//! - Performance status:
//!   + Performance of an operation depends on the memory layout of the array
//!     or array view. Especially if it's a binary operation, which
//!     needs matching memory layout to be efficient (with some exceptions).
//!   + Arithmetic optimizes very well if the arrays are have contiguous inner dimension.
//!   + The higher order functions like ``.map()``, ``.map_inplace()`` and
//!     ``.zip_mut_with()`` are the most efficient ways to
//!     perform single traversal and lock step traversal respectively.
//!   + ``.iter()`` is efficient for c-contiguous arrays.
//!   + Can use BLAS in some operations (`dot` and `mat_mul`).
//!
//! ## Crate Feature Flags
//!
//! The following crate feature flags are available. They are configured in your
//! `Cargo.toml`.
//!
//! - `rustc-serialize`
//!   - Optional, compatible with Rust stable
//!   - Enables serialization support
//! - `blas`
//!   - Optional and experimental, compatible with Rust stable
//!   - Enable transparent BLAS support for matrix multiplication. Pluggable
//!     backend via `blas-sys`.
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

use std::rc::Rc;
use std::slice::{self, Iter, IterMut};
use std::marker::PhantomData;

use itertools::ZipSlices;

pub use dimension::{
    Dimension,
    RemoveAxis,
    Axis,
};

pub use dimension::NdIndex;
pub use indexes::Indexes;
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
};

pub use arraytraits::AsArray;
pub use linalg_traits::{LinalgScalar, NdFloat};
pub use stacking::stack;

pub use shape_builder::{ ShapeBuilder };

mod arraytraits;
#[cfg(feature = "serde")]
mod arrayserialize;
mod arrayformat;
mod data_traits;

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
/// Type aliases [`OwnedArray`], [`RcArray`], [`ArrayView`], and [`ArrayViewMut`] refer
/// to `ArrayBase` with different types for the data container.
///
/// [`OwnedArray`]: type.OwnedArray.html
/// [`RcArray`]: type.RcArray.html
/// [`ArrayView`]: type.ArrayView.html
/// [`ArrayViewMut`]: type.ArrayViewMut.html
///
/// ## Contents
///
/// + [OwnedArray and RcArray](#ownedarray-and-rcarray)
/// + [Indexing and Dimension](#indexing-and-dimension)
/// + [Slicing](#slicing)
/// + [Subviews](#subviews)
/// + [Arithmetic Operations](#arithmetic-operations)
/// + [Broadcasting](#broadcasting)
/// + [Methods](#methods)
/// + [Methods for Array Views](#methods-for-array-views)
///
/// ## `OwnedArray` and `RcArray`
///
/// `OwnedArray` owns the underlying array elements directly (just like
/// a `Vec`), while [`RcArray`](type.RcArray.html) is a an array with reference
/// counted data. `RcArray` can act both as an owner or as a view in that regard.
/// Sharing requires that it uses copy-on-write for mutable operations.
/// Calling a method for mutating elements on `RcArray`, for example
/// [`view_mut()`](#method.view_mut) or [`get_mut()`](#method.get_mut),
/// will break sharing and require a clone of the data (if it is not uniquely held).
///
/// Note that all `ArrayBase` variants can change their view (slicing) of the
/// data freely, even when their data can’t be mutated.
///
/// ## Indexing and Dimension
///
/// Array indexes are represented by the types `Ix` and `Ixs` (signed).
///
/// The dimensionality of the array determines the number of *axes*, for example
/// a 2D array has two axes. These are listed in “big endian” order, so that
/// the greatest dimension is listed first, the lowest dimension with the most
/// rapidly varying index is the last.
///
/// In a 2D array the index of each element is `(row, column)`
/// as seen in this 3 × 3 example:
///
/// ```ignore
/// [[ (0, 0), (0, 1), (0, 2)],  // row 0
///  [ (1, 0), (1, 1), (1, 2)],  // row 1
///  [ (2, 0), (2, 1), (2, 2)]]  // row 2
/// //    \       \       \
/// //   column 0  \     column 2
/// //            column 1
/// ```
///
/// The number of axes for an array is fixed by the `D` parameter: `Ix` for
/// a 1D array, `(Ix, Ix)` for a 2D array etc. The `D` type is also used
/// for element indices in `.get()` and `array[index]`. The dimension type `Vec<Ix>`
/// allows a dynamic number of axes.
///
/// The default memory order of an array is *row major* order (a.k.a “c” order),
/// where each row is contiguous in memory.
/// A *column major* (a.k.a. “f” or fortran) memory order array has
/// columns (or, in general, the outermost axis) with contiguous elements.
///
/// The logical order of any array’s elements is the row major order.
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
/// with owned storage (either `OwnedArray` or `RcArray`).
/// Let `C` be an array with mutable data (either `OwnedArray`, `RcArray`
/// or `ArrayViewMut`).
/// The following combinations of operands
/// are supported for an arbitrary binary operator denoted by `@` (it can be
/// `+`, `-`, `*`, `/` and so on).
///
/// - `&A @ &A` which produces a new `OwnedArray`
/// - `B @ A` which consumes `B`, updates it with the result, and returns it
/// - `B @ &A` which consumes `B`, updates it with the result, and returns it
/// - `C @= &A` which performs an arithmetic operation in place
///
/// The trait [`ScalarOperand`](trait.ScalarOperand.html) marks types that can be used in arithmetic
/// with arrays directly. For a scalar `K` the following combinations of operands
/// are supported (scalar can be on either the left or right side, but
/// `ScalarOperand` docs has the detailed condtions).
///
/// - `&A @ K` or `K @ &A` which produces a new `OwnedArray`
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
///                [1., 2.]]);
/// let b = arr2(&[[0., 1.]]);
///
/// let c = arr2(&[[1., 2.],
///                [1., 3.]]);
/// // We can add because the shapes are compatible even if not equal.
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

/// Array where the data is reference counted and copy on write, it
/// can act as both an owner as the data as well as a lightweight view.
pub type RcArray<A, D> = ArrayBase<Rc<Vec<A>>, D>;

/// Array where the data is owned uniquely.
pub type OwnedArray<A, D> = ArrayBase<Vec<A>, D>;

/// A lightweight array view.
///
/// An array view represents an array or a part of it, created from
/// an iterator, subview or slice of an array.
///
/// Array views have all the methods of an array (see [`ArrayBase`][ab]).
///
/// See also specific [**Methods for Array Views**](struct.ArrayBase.html#methods-for-array-views).
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
/// See also specific [**Methods for Array Views**](struct.ArrayBase.html#methods-for-array-views).
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
        for mut row in self.inner_iter_mut() {
            if let Some(slc) = row.as_slice_mut() {
                // FIXME: Use for loop when slice iterator is perf is restored
                for i in 0..slc.len() {
                    f(&mut slc[i]);
                }
                continue;
            }
            for elt in row {
                f(elt);
            }
        }
    }
}


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
            Baseiter::new(self.ptr, self.dim.clone(), self.strides.clone())
        }
    }

    #[inline]
    fn into_elements_base(self) -> ElementsBase<'a, A, D> {
        ElementsBase { inner: self.into_base_iter() }
    }

    fn into_iter_(self) -> Elements<'a, A, D> {
        Elements {
            inner: if let Some(slc) = self.into_slice() {
                ElementsRepr::Slice(slc.iter())
            } else {
                ElementsRepr::Counted(self.into_elements_base())
            },
        }
    }

    fn into_slice(&self) -> Option<&'a [A]> {
        if self.is_standard_layout() {
            unsafe {
                Some(slice::from_raw_parts(self.ptr, self.len()))
            }
        } else {
            None
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
            Baseiter::new(self.ptr, self.dim.clone(), self.strides.clone())
        }
    }

    #[inline]
    fn into_elements_base(self) -> ElementsBaseMut<'a, A, D> {
        ElementsBaseMut { inner: self.into_base_iter() }
    }

    fn into_iter_(self) -> ElementsMut<'a, A, D> {
        ElementsMut {
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

    fn _into_slice_mut(self) -> Option<&'a mut [A]>
    {
        if self.is_standard_layout() {
            unsafe {
                Some(slice::from_raw_parts_mut(self.ptr, self.len()))
            }
        } else {
            None
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
pub struct Elements<'a, A: 'a, D> {
    inner: ElementsRepr<Iter<'a, A>, ElementsBase<'a, A, D>>,
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
pub struct ElementsMut<'a, A: 'a, D> {
    inner: ElementsRepr<IterMut<'a, A>, ElementsBaseMut<'a, A, D>>,
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
pub struct Indexed<'a, A: 'a, D>(ElementsBase<'a, A, D>);
/// An iterator over the indexes and elements of an array (mutable).
///
/// See [`.indexed_iter_mut()`](struct.ArrayBase.html#method.indexed_iter_mut) for more information.
pub struct IndexedMut<'a, A: 'a, D>(ElementsBaseMut<'a, A, D>);

fn zipsl<T, U>(t: T, u: U) -> ZipSlices<T, U>
    where T: itertools::misc::Slice, U: itertools::misc::Slice
{
    ZipSlices::from_slices(t, u)
}

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

/// An array shape of n dimensions c-order, f-order or custom strides.
#[derive(Copy, Clone, Debug)]
pub struct StrideShape<D> {
    dim: D,
    strides: D,
    custom: bool,
}
