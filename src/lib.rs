#![crate_name="ndarray"]
#![cfg_attr(has_deprecated, feature(deprecated))]
#![doc(html_root_url = "http://bluss.github.io/rust-ndarray/master/")]

//! The `ndarray` crate provides an N-dimensional container similar to numpy’s
//! ndarray.
//!
//! - [`ArrayBase`](struct.ArrayBase.html):
//!   The N-dimensional array type itself.
//! - [`OwnedArray`](type.OwnedArray.html):
//!   An array where the data is owned uniquely.
//! - [`RcArray`](type.RcArray.html):
//!   An array where the data is shared and copy on write, it
//!   can act as both an owner of the data as well as a lightweight view.
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
//!   `aview1` and `aview_mut1`.
//!
//! ## Crate Status
//!
//! - Still iterating on and evolving the API
//!   + The crate is continuously developing, and breaking changes are expected
//!     during evolution from version to version. We adhere to semver,
//!     but alpha releases break at will.
//!   + We adopt the newest stable rust features we need. In place methods like `iadd`
//!     *will be deprecated* when Rust supports `+=` and similar in Rust 1.8.
//!   + We try to introduce more static checking gradually.
//! - Performance status:
//!   + Arithmetic involving arrays of contiguous inner dimension optimizes very well.
//!   + `.fold()` and `.zip_mut_with()` are the most efficient ways to
//!     perform single traversal and lock step traversal respectively.
//!   + `.iter()` and `.iter_mut()` are efficient for contiguous arrays.
//! - There is experimental bridging to the linear algebra package `rblas`.
//!
//! ## Crate Feature Flags
//!
//! The following crate feature flags are available. The are specified in
//! `Cargo.toml`.
//!
//! - `assign_ops`
//!   - Optional, requires nightly
//!   - Enables the compound assignment operators
//! - `rustc-serialize`
//!   - Optional, stable
//!   - Enables serialization support
//! - `rblas`
//!   - Optional, stable
//!   - Enables `rblas` integration
//!
#![cfg_attr(feature = "assign_ops", feature(augmented_assignments,
                                            op_assign_traits))]

#[cfg(feature = "serde")]
extern crate serde;
#[cfg(feature = "rustc-serialize")]
extern crate rustc_serialize as serialize;

#[cfg(feature = "rblas")]
extern crate rblas;

extern crate itertools;
extern crate num as libnum;

use libnum::Float;
use libnum::Complex;

use std::cmp;
use std::mem;
use std::ops::{Add, Sub, Mul, Div, Rem, Neg, Not, Shr, Shl,
    BitAnd,
    BitOr,
    BitXor,
};
use std::rc::Rc;
use std::slice::{self, Iter, IterMut};
use std::marker::PhantomData;

use itertools::ZipSlices;
use itertools::free::enumerate;

pub use dimension::{
    Dimension,
    RemoveAxis,
    Axis,
};

pub use dimension::NdIndex;
pub use indexes::Indexes;
pub use shape_error::ShapeError;
pub use stride_error::StrideError;
pub use si::{Si, S};

use iterators::Baseiter;
pub use iterators::{
    InnerIter,
    InnerIterMut,
    OuterIter,
    OuterIterMut,
    AxisChunksIter,
    AxisChunksIterMut,
};

pub use linalg::LinalgScalar;

mod arraytraits;
#[cfg(feature = "serde")]
mod arrayserialize;
mod arrayformat;
#[cfg(feature = "rblas")]
pub mod blas;
mod dimension;
mod indexes;
mod iterators;
mod linalg;
mod linspace;
mod numeric_util;
mod si;
mod shape_error;
mod stride_error;

/// Implementation's prelude. Common types used everywhere.
mod imp_prelude {
    pub use {
        ArrayBase,
        ArrayView,
        ArrayViewMut,
        OwnedArray,
        RcArray,
        Ix, Ixs,
        Dimension,
        Data,
        DataMut,
        DataOwned,
    };
    /// Wrapper type for private methods
    #[derive(Copy, Clone, Debug)]
    pub struct Priv<T>(pub T);
}

// NOTE: In theory, the whole library should compile
// and pass tests even if you change Ix and Ixs.
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
/// The `ArrayBase<S, D>` is parameterized by:

/// - `S` for the data container
/// - `D` for the number of dimensions
///
/// Type aliases [`OwnedArray`], [`RcArray`], [`ArrayView`], and [`ArrayViewMut`] refer
/// to `ArrayBase` with different types for the data storage.
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
/// The default memory order of an array is *row major* order, where each
/// row is contiguous in memory.
/// A *column major* (a.k.a. fortran) memory order array has
/// columns (or, in general, the outermost axis) with contiguous elements.
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
/// are supported for an arbitrary binary operator denoted by `@`.
///
/// - `&A @ &A` which produces a new `OwnedArray`
/// - `B @ A` which consumes `B`, updates it with the result, and returns it
/// - `B @ &A` which consumes `B`, updates it with the result, and returns it
/// - `C @= &A` which performs an arithmetic operation in place
///   (requires crate feature `"assign_ops"`)
///
/// The trait [`Scalar`](trait.Scalar.html) marks types that can be used in arithmetic
/// with arrays directly. For a scalar `K` the following combinations of operands
/// are supported (scalar can be on either side).
///
/// - `&A @ K` or `K @ &A` which produces a new `OwnedArray`
/// - `B @ K` or `K @ B` which consumes `B`, updates it with the result and returns it
/// - `C @= K` which performs an arithmetic operation in place
///   (requires crate feature `"assign_ops"`)
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

/// Array’s inner representation.
///
/// ***Note:*** `Data` is not an extension interface at this point.
/// Traits in Rust can serve many different roles. This trait is public because
/// it is used as a bound on public methods.
pub unsafe trait Data {
    type Elem;
    fn slice(&self) -> &[Self::Elem];
}

/// Array’s writable inner representation.
pub unsafe trait DataMut : Data {
    fn slice_mut(&mut self) -> &mut [Self::Elem];
    #[inline]
    fn ensure_unique<D>(&mut ArrayBase<Self, D>)
        where Self: Sized,
              D: Dimension
    { }

    #[inline]
    fn is_unique(&mut self) -> bool {
        true
    }
}

/// Clone an Array’s storage.
pub unsafe trait DataClone : Data {
    /// Unsafe because, `ptr` must point inside the current storage.
    unsafe fn clone_with_ptr(&self, ptr: *mut Self::Elem) -> (Self, *mut Self::Elem);
}

unsafe impl<A> Data for Rc<Vec<A>> {
    type Elem = A;
    fn slice(&self) -> &[A] {
        self
    }
}

// NOTE: Copy on write
unsafe impl<A> DataMut for Rc<Vec<A>>
    where A: Clone
{
    fn slice_mut(&mut self) -> &mut [A] {
        &mut Rc::make_mut(self)[..]
    }

    fn ensure_unique<D>(self_: &mut ArrayBase<Self, D>)
        where Self: Sized,
              D: Dimension
    {
        if Rc::get_mut(&mut self_.data).is_some() {
            return;
        }
        if self_.dim.size() <= self_.data.len() / 2 {
            // Create a new vec if the current view is less than half of
            // backing data.
            unsafe {
                *self_ = ArrayBase::from_vec_dim_unchecked(self_.dim.clone(),
                                                           self_.iter()
                                                            .cloned()
                                                            .collect());
            }
            return;
        }
        let our_off = (self_.ptr as isize - self_.data.as_ptr() as isize) /
                      mem::size_of::<A>() as isize;
        let rvec = Rc::make_mut(&mut self_.data);
        unsafe {
            self_.ptr = rvec.as_mut_ptr().offset(our_off);
        }
    }

    fn is_unique(&mut self) -> bool {
        Rc::get_mut(self).is_some()
    }
}

unsafe impl<A> DataClone for Rc<Vec<A>> {
    unsafe fn clone_with_ptr(&self, ptr: *mut Self::Elem) -> (Self, *mut Self::Elem) {
        // pointer is preserved
        (self.clone(), ptr)
    }
}

unsafe impl<A> Data for Vec<A> {
    type Elem = A;
    fn slice(&self) -> &[A] {
        self
    }
}

unsafe impl<A> DataMut for Vec<A> {
    fn slice_mut(&mut self) -> &mut [A] {
        self
    }
}

unsafe impl<A> DataClone for Vec<A>
    where A: Clone
{
    unsafe fn clone_with_ptr(&self, ptr: *mut Self::Elem) -> (Self, *mut Self::Elem) {
        let mut u = self.clone();
        let our_off = (self.as_ptr() as isize - ptr as isize) /
                      mem::size_of::<A>() as isize;
        let new_ptr = u.as_mut_ptr().offset(our_off);
        (u, new_ptr)
    }
}

unsafe impl<'a, A> Data for ViewRepr<&'a A> {
    type Elem = A;
    fn slice(&self) -> &[A] {
        &[]
    }
}

unsafe impl<'a, A> DataClone for ViewRepr<&'a A> {
    unsafe fn clone_with_ptr(&self, ptr: *mut Self::Elem) -> (Self, *mut Self::Elem) {
        (*self, ptr)
    }
}

unsafe impl<'a, A> Data for ViewRepr<&'a mut A> {
    type Elem = A;
    fn slice(&self) -> &[A] {
        &[]
    }
}

unsafe impl<'a, A> DataMut for ViewRepr<&'a mut A> {
    fn slice_mut(&mut self) -> &mut [A] {
        &mut []
    }
}

/// Array representation that is a unique or shared owner of its data.
pub unsafe trait DataOwned : Data {
    fn new(elements: Vec<Self::Elem>) -> Self;
    fn into_shared(self) -> Rc<Vec<Self::Elem>>;
}

/// Array representation that is a lightweight view.
pub unsafe trait DataShared : Clone + DataClone { }

unsafe impl<A> DataShared for Rc<Vec<A>> {}
unsafe impl<'a, A> DataShared for ViewRepr<&'a A> {}

unsafe impl<A> DataOwned for Vec<A> {
    fn new(elements: Vec<A>) -> Self {
        elements
    }
    fn into_shared(self) -> Rc<Vec<A>> {
        Rc::new(self)
    }
}

unsafe impl<A> DataOwned for Rc<Vec<A>> {
    fn new(elements: Vec<A>) -> Self {
        Rc::new(elements)
    }
    fn into_shared(self) -> Rc<Vec<A>> {
        self
    }
}


/// Array where the data is reference counted and copy on write, it
/// can act as both an owner as the data as well as a lightweight view.
pub type RcArray<A, D> = ArrayBase<Rc<Vec<A>>, D>;

#[cfg_attr(has_deprecated, deprecated(note="`Array` is deprecated! Renamed to `RcArray`."))]
/// ***Deprecated: Use `RcArray` instead***
///
/// Array where the data is reference counted and copy on write, it
/// can act as both an owner as the data as well as a lightweight view.
pub type Array<A, D> = ArrayBase<Rc<Vec<A>>, D>;

/// Array where the data is owned uniquely.
pub type OwnedArray<A, D> = ArrayBase<Vec<A>, D>;

/// A lightweight array view.
///
/// `ArrayView` implements `IntoIterator`.
pub type ArrayView<'a, A, D> = ArrayBase<ViewRepr<&'a A>, D>;
/// A lightweight read-write array view.
///
/// `ArrayViewMut` implements `IntoIterator`.
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

impl<S: DataClone, D: Clone> Clone for ArrayBase<S, D> {
    fn clone(&self) -> ArrayBase<S, D> {
        unsafe {
            let (data, ptr) = self.data.clone_with_ptr(self.ptr);
            ArrayBase {
                data: data,
                ptr: ptr,
                dim: self.dim.clone(),
                strides: self.strides.clone(),
            }
        }
    }
}

impl<S: DataClone + Copy, D: Copy> Copy for ArrayBase<S, D> {}

/// Constructor methods for one-dimensional arrays.
impl<S> ArrayBase<S, Ix>
    where S: DataOwned
{
    /// Create a one-dimensional array from a vector (no allocation needed).
    pub fn from_vec(v: Vec<S::Elem>) -> ArrayBase<S, Ix> {
        unsafe { Self::from_vec_dim_unchecked(v.len() as Ix, v) }
    }

    /// Create a one-dimensional array from an iterable.
    pub fn from_iter<I: IntoIterator<Item=S::Elem>>(iterable: I) -> ArrayBase<S, Ix> {
        Self::from_vec(iterable.into_iter().collect())
    }

    /// Create a one-dimensional array from inclusive interval
    /// `[start, end]` with `n` elements. `F` must be a floating point type.
    pub fn linspace<F>(start: F, end: F, n: usize) -> ArrayBase<S, Ix>
        where S: Data<Elem=F>,
              F: libnum::Float,
    {
        Self::from_iter(linspace::linspace(start, end, n))
    }
}

/// Constructor methods for two-dimensional arrays.
impl<S, A> ArrayBase<S, (Ix, Ix)>
    where S: DataOwned<Elem=A>,
{
    /// Create an identity matrix of size `n` (square 2D array).
    ///
    /// **Panics** if `n * n` would overflow usize.
    pub fn eye(n: Ix) -> ArrayBase<S, (Ix, Ix)>
        where S: DataMut,
              A: Clone + libnum::Zero + libnum::One,
    {
        let mut eye = Self::zeros((n, n));
        for a_ii in eye.diag_mut() {
            *a_ii = A::one();
        }
        eye
    }
}

/// Constructor methods for arrays.
impl<S, A, D> ArrayBase<S, D>
    where S: DataOwned<Elem=A>,
          D: Dimension,
{
    /// Create an array with copies of `elem`, dimension `dim`.
    ///
    /// **Panics** if the number of elements in `dim` would overflow usize.
    ///
    /// ```
    /// use ndarray::RcArray;
    /// use ndarray::arr3;
    ///
    /// let a = RcArray::from_elem((2, 2, 2), 1.);
    ///
    /// assert!(
    ///     a == arr3(&[[[1., 1.],
    ///                  [1., 1.]],
    ///                 [[1., 1.],
    ///                  [1., 1.]]])
    /// );
    /// ```
    pub fn from_elem(dim: D, elem: A) -> ArrayBase<S, D>
        where A: Clone
    {
        // Note: We don't need to check the case of a size between
        // isize::MAX -> usize::MAX; in this case, the vec constructor itself
        // panics.
        let size = dim.size_checked().expect("Shape too large: overflow in size");
        let v = vec![elem; size];
        unsafe { Self::from_vec_dim_unchecked(dim, v) }
    }

    /// Create an array with copies of `elem`, dimension `dim` and fortran
    /// memory order.
    ///
    /// **Panics** if the number of elements would overflow usize.
    ///
    /// ```
    /// use ndarray::RcArray;
    /// use ndarray::arr3;
    ///
    /// let a = RcArray::from_elem_f((2, 2, 2), 1.);
    ///
    /// assert!(
    ///     a == arr3(&[[[1., 1.],
    ///                  [1., 1.]],
    ///                 [[1., 1.],
    ///                  [1., 1.]]])
    /// );
    /// assert!(a.strides() == &[1, 2, 4]);
    /// ```
    pub fn from_elem_f(dim: D, elem: A) -> ArrayBase<S, D>
        where A: Clone
    {
        let size = dim.size_checked().expect("Shape too large: overflow in size");
        let v = vec![elem; size];
        unsafe { Self::from_vec_dim_unchecked_f(dim, v) }
    }

    /// Create an array with zeros, dimension `dim`.
    ///
    /// **Panics** if the number of elements in `dim` would overflow usize.
    pub fn zeros(dim: D) -> ArrayBase<S, D>
        where A: Clone + libnum::Zero
    {
        Self::from_elem(dim, libnum::zero())
    }

    /// Create an array with zeros, dimension `dim` and fortran memory order.
    ///
    /// **Panics** if the number of elements in `dim` would overflow usize.
    pub fn zeros_f(dim: D) -> ArrayBase<S, D>
        where A: Clone + libnum::Zero
    {
        Self::from_elem_f(dim, libnum::zero())
    }

    /// Create an array with default values, dimension `dim`.
    ///
    /// **Panics** if the number of elements in `dim` would overflow usize.
    pub fn default(dim: D) -> ArrayBase<S, D>
        where A: Default
    {
        let v = (0..dim.size()).map(|_| A::default()).collect();
        unsafe { Self::from_vec_dim_unchecked(dim, v) }
    }

    /// Create an array from a vector (with no allocation needed).
    ///
    /// **Errors** if `dim` does not correspond to the number of elements
    /// in `v`.
    pub fn from_vec_dim(dim: D, v: Vec<A>) -> Result<ArrayBase<S, D>, ShapeError> {
        if dim.size_checked() != Some(v.len()) {
            return Err(shape_error::incompatible_shapes(&v.len(), &dim));
        }
        unsafe { Ok(Self::from_vec_dim_unchecked(dim, v)) }
    }

    /// Create an array from a vector (with no allocation needed).
    ///
    /// Unsafe because dimension is unchecked, and must be correct.
    pub unsafe fn from_vec_dim_unchecked(dim: D, mut v: Vec<A>) -> ArrayBase<S, D> {
        debug_assert!(dim.size_checked() == Some(v.len()));
        ArrayBase {
            ptr: v.as_mut_ptr(),
            data: DataOwned::new(v),
            strides: dim.default_strides(),
            dim: dim,
        }
    }

    /// Create an array from a vector (with no allocation needed),
    /// using fortran memory order to interpret the data.
    ///
    /// Unsafe because dimension is unchecked, and must be correct.
    pub unsafe fn from_vec_dim_unchecked_f(dim: D, mut v: Vec<A>) -> ArrayBase<S, D> {
        debug_assert!(dim.size_checked() == Some(v.len()));
        ArrayBase {
            ptr: v.as_mut_ptr(),
            data: DataOwned::new(v),
            strides: dim.fortran_strides(),
            dim: dim,
        }
    }

    /// Create an array from a vector and interpret it according to the
    /// provided dimensions and strides. No allocation needed.
    ///
    /// Checks whether `dim` and `strides` are compatible with the vector's
    /// length, returning an `Err` if not compatible.
    ///
    /// **Errors** if strides and dimensions can point out of bounds of `v`.<br>
    /// **Errors** if strides allow multiple indices to point to the same element.
    pub fn from_vec_dim_stride(dim: D, strides: D, v: Vec<A>)
        -> Result<ArrayBase<S, D>, StrideError>
    {
        dimension::can_index_slice(&v, &dim, &strides).map(|_| {
            unsafe {
                Self::from_vec_dim_stride_unchecked(dim, strides, v)
            }
        })
    }

    /// Create an array from a vector and interpret it according to the
    /// provided dimensions and strides. No allocation needed.
    ///
    /// Unsafe because dimension and strides are unchecked.
    pub unsafe fn from_vec_dim_stride_unchecked(dim: D, strides: D, mut v: Vec<A>)
        -> ArrayBase<S, D>
    {
        debug_assert!(dimension::can_index_slice(&v, &dim, &strides).is_ok());
        ArrayBase {
            ptr: v.as_mut_ptr(),
            data: DataOwned::new(v),
            strides: strides,
            dim: dim
        }
    }

}


// ArrayView methods
impl<'a, A> ArrayView<'a, A, Ix> {
    #[inline]
    fn from_slice(xs: &'a [A]) -> Self {
        ArrayView {
            data: ViewRepr::new(),
            ptr: xs.as_ptr() as *mut A,
            dim: xs.len(),
            strides: 1,
        }
    }
}

impl<'a, A, D> ArrayView<'a, A, D>
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

    /// Create an `ArrayView` borrowing its data from a slice.
    ///
    /// Checks whether `dim` and `strides` are compatible with the slice's
    /// length, returning an `Err` if not compatible.
    ///
    /// ```
    /// use ndarray::ArrayView;
    /// use ndarray::arr3;
    ///
    /// let s = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    /// let a = ArrayView::from_slice_dim_stride((2, 3, 2),
    ///                                          (1, 4, 2),
    ///                                          s).unwrap();
    ///
    /// assert!(
    ///     a == arr3(&[[[0, 2],
    ///                  [4, 6],
    ///                  [8, 10]],
    ///                 [[1, 3],
    ///                  [5, 7],
    ///                  [9, 11]]])
    /// );
    /// assert!(a.strides() == &[1, 4, 2]);
    /// ```
    pub fn from_slice_dim_stride(dim: D, strides: D, s: &'a [A])
        -> Result<Self, StrideError>
    {
        dimension::can_index_slice(s, &dim, &strides).map(|_| {
            unsafe {
                Self::new_(s.as_ptr(), dim, strides)
            }
        })
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
    pub fn into_outer_iter(self) -> OuterIter<'a, A, D::Smaller>
        where D: RemoveAxis,
    {
        iterators::new_outer_iter(self)
    }
}

impl<'a, A, D> ArrayViewMut<'a, A, D>
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

    /// Create an `ArrayView` borrowing its data from a slice.
    ///
    /// Checks whether `dim` and `strides` are compatible with the slice's
    /// length, returning an `Err` if not compatible.
    ///
    /// ```
    /// use ndarray::ArrayViewMut;
    /// use ndarray::arr3;
    ///
    /// let s = &mut [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    /// let mut a = ArrayViewMut::from_slice_dim_stride((2, 3, 2),
    ///                                                 (1, 4, 2),
    ///                                                 s).unwrap();
    ///
    /// a[[0, 0, 0]] = 1;
    /// assert!(
    ///     a == arr3(&[[[1, 2],
    ///                  [4, 6],
    ///                  [8, 10]],
    ///                 [[1, 3],
    ///                  [5, 7],
    ///                  [9, 11]]])
    /// );
    /// assert!(a.strides() == &[1, 4, 2]);
    /// ```
    pub fn from_slice_dim_stride(dim: D, strides: D, s: &'a mut [A])
        -> Result<Self, StrideError>
    {
        dimension::can_index_slice(s, &dim, &strides).map(|_| {
            unsafe {
                Self::new_(s.as_mut_ptr(), dim, strides)
            }
        })
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

    pub fn into_outer_iter(self) -> OuterIterMut<'a, A, D::Smaller>
        where D: RemoveAxis,
    {
        iterators::new_outer_iter_mut(self)
    }
}

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
        debug_assert!(self.dim
                          .stride_offset_checked(&self.strides, &index)
                          .is_some());
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
        debug_assert!(self.dim
                          .stride_offset_checked(&self.strides, &index)
                          .is_some());
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
    /// use ndarray::{arr1, arr2, Axis};
    ///
    /// let a = arr2(&[[1., 2.],    // -- axis 0, row 0
    ///                [3., 4.],    // -- axis 0, row 1
    ///                [5., 6.]]);  // -- axis 0, row 2
    /// //               \   \
    /// //                \   axis 1, column 1
    /// //                 axis 1, column 0
    /// assert!(
    ///     a.subview(Axis(0), 1) == arr1(&[3., 4.]) &&
    ///     a.subview(Axis(1), 1) == arr1(&[2., 4., 6.])
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
        let axis = axis.axis();
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
    pub fn outer_iter(&self) -> OuterIter<A, D::Smaller>
        where D: RemoveAxis,
    {
        iterators::new_outer_iter(self.view())
    }

    /// Return an iterator that traverses over the outermost dimension
    /// and yields each subview.
    ///
    /// Iterator element is `ArrayViewMut<A, D::Smaller>` (read-write array view).
    pub fn outer_iter_mut(&mut self) -> OuterIterMut<A, D::Smaller>
        where S: DataMut,
              D: RemoveAxis,
    {
        iterators::new_outer_iter_mut(self.view_mut())
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

    #[cfg(feature = "rblas")]
    /// If the array is not in the standard layout, copy all elements
    /// into the standard layout so that the array is C-contiguous.
    fn ensure_standard_layout(&mut self)
        where S: DataOwned,
              A: Clone
    {
        if !self.is_standard_layout() {
            let mut v: Vec<A> = self.iter().cloned().collect();
            self.ptr = v.as_mut_ptr();
            self.data = DataOwned::new(v);
            self.strides = self.dim.default_strides();
        }
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

    #[cfg(feature = "rblas")]
    /// Return `true` if the innermost dimension is contiguous (includes
    /// the special cases of 0 or 1 length in that axis).
    fn is_inner_contiguous(&self) -> bool {
        let ndim = self.ndim();
        if ndim == 0 {
            return true;
        }
        self.shape()[ndim - 1] <= 1 || self.strides()[ndim - 1] == 1
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
            panic!("Incompatible shapes in reshape, attempted from: {:?}, to: {:?}",
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
            return Err(shape_error::incompatible_shapes(&self.dim, &shape));
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
            Err(ShapeError::IncompatibleLayout)
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
            panic!("Could not broadcast array from shape: {:?} to: {:?}",
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
    /// **Note:** the slice may be empty.
    pub fn raw_data(&self) -> &[A] {
        self.data.slice()
    }

    /// Return a mutable slice of the array’s backing data in memory order.
    ///
    /// **Note:** Data memory order may not correspond to the index order
    /// of the array. Neither is the raw data slice is restricted to just the
    /// array’s view.<br>
    /// **Note:** the slice may be empty.
    ///
    /// **Note:** The data is uniquely held and nonaliased
    /// while it is mutably borrowed.
    pub fn raw_data_mut(&mut self) -> &mut [A]
        where S: DataMut,
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

    /// Apply closure `f` to each element in the array, in whatever
    /// order is the fastest to visit.
    fn unordered_foreach_mut<F>(&mut self, mut f: F)
        where S: DataMut,
              F: FnMut(&mut A)
    {
        if let Some(slc) = self.as_slice_mut() {
            for elt in slc {
                f(elt);
            }
            return;
        }
        for row in self.inner_iter_mut() {
            for elt in row {
                f(elt);
            }
        }
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
        let mut res = Vec::with_capacity(self.dim.size());
        for elt in self.iter() {
            res.push(f(elt))
        }
        unsafe {
            ArrayBase::from_vec_dim_unchecked(self.dim.clone(), res)
        }
    }
}

/// ***Deprecated: Use `ArrayBase::zeros` instead.***
///
/// Return an array filled with zeros
#[cfg_attr(has_deprecated, deprecated(note="Use `ArrayBase::zeros` instead."))]
pub fn zeros<A, D>(dim: D) -> OwnedArray<A, D>
    where A: Clone + libnum::Zero, D: Dimension,
{
    ArrayBase::zeros(dim)
}

/// Return a zero-dimensional array with the element `x`.
pub fn arr0<A>(x: A) -> OwnedArray<A, ()>
{
    unsafe { ArrayBase::from_vec_dim_unchecked((), vec![x]) }
}

/// Return a one-dimensional array with elements from `xs`.
pub fn arr1<A: Clone>(xs: &[A]) -> OwnedArray<A, Ix> {
    ArrayBase::from_vec(xs.to_vec())
}

/// Return a one-dimensional array with elements from `xs`.
pub fn rcarr1<A: Clone>(xs: &[A]) -> RcArray<A, Ix> {
    arr1(xs).into_shared()
}

/// Return a zero-dimensional array view borrowing `x`.
pub fn aview0<A>(x: &A) -> ArrayView<A, ()> {
    unsafe { ArrayView::new_(x, (), ()) }
}

/// Return a one-dimensional array view with elements borrowing `xs`.
///
/// ```
/// use ndarray::aview1;
///
/// let data = [1.0; 1024];
///
/// // Create a 2D array view from borrowed data
/// let a2d = aview1(&data).into_shape((32, 32)).unwrap();
///
/// assert!(
///     a2d.scalar_sum() == 1024.0
/// );
/// ```
pub fn aview1<A>(xs: &[A]) -> ArrayView<A, Ix> {
    ArrayView::from_slice(xs)
}

/// Return a two-dimensional array view with elements borrowing `xs`.
pub fn aview2<A, V: FixedInitializer<Elem=A>>(xs: &[V]) -> ArrayView<A, (Ix, Ix)> {
    let cols = V::len();
    let rows = xs.len();
    let data = unsafe {
        std::slice::from_raw_parts(xs.as_ptr() as *const A, cols * rows)
    };
    let dim = (rows as Ix, cols as Ix);
    unsafe {
        let strides = dim.default_strides();
        ArrayView::new_(data.as_ptr(), dim, strides)
    }
}

/// Return a one-dimensional read-write array view with elements borrowing `xs`.
///
/// ```
/// #[macro_use(s)]
/// extern crate ndarray;
///
/// use ndarray::aview_mut1;
///
/// // Create an array view over some data, then slice it and modify it.
/// fn main() {
///     let mut data = [0; 1024];
///     {
///         let mut a = aview_mut1(&mut data).into_shape((32, 32)).unwrap();
///         a.slice_mut(s![.., ..;3]).assign_scalar(&5);
///     }
///     assert_eq!(&data[..10], [5, 0, 0, 5, 0, 0, 5, 0, 0, 5]);
/// }
/// ```
pub fn aview_mut1<A>(xs: &mut [A]) -> ArrayViewMut<A, Ix> {
    unsafe { ArrayViewMut::new_(xs.as_mut_ptr(), xs.len() as Ix, 1) }
}

/// Fixed-size array used for array initialization
pub unsafe trait FixedInitializer {
    type Elem;
    fn as_init_slice(&self) -> &[Self::Elem];
    fn len() -> usize;
}

macro_rules! impl_arr_init {
    (__impl $n: expr) => (
        unsafe impl<T> FixedInitializer for [T;  $n] {
            type Elem = T;
            fn as_init_slice(&self) -> &[T] { self }
            fn len() -> usize { $n }
        }
    );
    () => ();
    ($n: expr, $($m:expr,)*) => (
        impl_arr_init!(__impl $n);
        impl_arr_init!($($m,)*);
    )

}

impl_arr_init!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,);

/// Return a two-dimensional array with elements from `xs`.
///
/// ```
/// use ndarray::arr2;
///
/// let a = arr2(&[[1, 2, 3],
///                [4, 5, 6]]);
/// assert!(
///     a.shape() == [2, 3]
/// );
/// ```
pub fn arr2<A: Clone, V: FixedInitializer<Elem = A>>(xs: &[V]) -> OwnedArray<A, (Ix, Ix)> {
    // FIXME: Simplify this when V is fix size array
    let (m, n) = (xs.len() as Ix,
                  xs.get(0).map_or(0, |snd| snd.as_init_slice().len() as Ix));
    let dim = (m, n);
    let mut result = Vec::<A>::with_capacity(dim.size());
    for snd in xs {
        let snd = snd.as_init_slice();
        result.extend(snd.iter().cloned());
    }
    unsafe {
        ArrayBase::from_vec_dim_unchecked(dim, result)
    }
}

/// Return a two-dimensional array with elements from `xs`.
///
pub fn rcarr2<A: Clone, V: FixedInitializer<Elem = A>>(xs: &[V]) -> RcArray<A, (Ix, Ix)> {
    arr2(xs).into_shared()
}

/// Return a three-dimensional array with elements from `xs`.
///
/// **Panics** if the slices are not all of the same length.
///
/// ```
/// use ndarray::arr3;
///
/// let a = arr3(&[[[1, 2],
///                 [3, 4]],
///                [[5, 6],
///                 [7, 8]],
///                [[9, 0],
///                 [1, 2]]]);
/// assert!(
///     a.shape() == [3, 2, 2]
/// );
/// ```
pub fn arr3<A: Clone, V: FixedInitializer<Elem=U>, U: FixedInitializer<Elem=A>>(xs: &[V])
    -> OwnedArray<A, (Ix, Ix, Ix)>
{
    // FIXME: Simplify this when U/V are fix size arrays
    let m = xs.len() as Ix;
    let fst = xs.get(0).map(|snd| snd.as_init_slice());
    let thr = fst.and_then(|elt| elt.get(0).map(|elt2| elt2.as_init_slice()));
    let n = fst.map_or(0, |v| v.len() as Ix);
    let o = thr.map_or(0, |v| v.len() as Ix);
    let dim = (m, n, o);
    let mut result = Vec::<A>::with_capacity(dim.size());
    for snd in xs {
        let snd = snd.as_init_slice();
        for thr in snd.iter() {
            let thr = thr.as_init_slice();
            result.extend(thr.iter().cloned());
        }
    }
    unsafe {
        ArrayBase::from_vec_dim_unchecked(dim, result)
    }
}

/// Return a three-dimensional array with elements from `xs`.
pub fn rcarr3<A: Clone, V: FixedInitializer<Elem=U>, U: FixedInitializer<Elem=A>>(xs: &[V])
    -> RcArray<A, (Ix, Ix, Ix)>
{
    arr3(xs).into_shared()
}

impl<A, S, D> ArrayBase<S, D>
    where S: Data<Elem=A>,
          D: Dimension,
{
    /// Return sum along `axis`.
    ///
    /// ```
    /// use ndarray::{aview0, aview1, arr2, Axis};
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert!(
    ///     a.sum(Axis(0)) == aview1(&[4., 6.]) &&
    ///     a.sum(Axis(1)) == aview1(&[3., 7.]) &&
    ///
    ///     a.sum(Axis(0)).sum(Axis(0)) == aview0(&10.)
    /// );
    /// ```
    ///
    /// **Panics** if `axis` is out of bounds.
    pub fn sum(&self, axis: Axis) -> OwnedArray<A, <D as RemoveAxis>::Smaller>
        where A: Clone + Add<Output=A>,
              D: RemoveAxis,
    {
        let n = self.shape()[axis.axis()];
        let mut res = self.subview(axis, 0).to_owned();
        for i in 1..n {
            let view = self.subview(axis, i);
            res.iadd(&view);
        }
        res
    }

    /// Return the sum of all elements in the array.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert_eq!(a.scalar_sum(), 10.);
    /// ```
    pub fn scalar_sum(&self) -> A
        where A: Clone + Add<Output=A> + libnum::Zero,
    {
        if let Some(slc) = self.as_slice() {
            return numeric_util::unrolled_sum(slc);
        }
        let mut sum = A::zero();
        for row in self.inner_iter() {
            if let Some(slc) = row.as_slice() {
                sum = sum + numeric_util::unrolled_sum(slc);
            } else {
                sum = sum + row.fold(A::zero(), |acc, elt| acc + elt.clone());
            }
        }
        sum
    }

    /// Return mean along `axis`.
    ///
    /// **Panics** if `axis` is out of bounds.
    ///
    /// ```
    /// use ndarray::{aview1, arr2, Axis};
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert!(
    ///     a.mean(Axis(0)) == aview1(&[2.0, 3.0]) &&
    ///     a.mean(Axis(1)) == aview1(&[1.5, 3.5])
    /// );
    /// ```
    pub fn mean(&self, axis: Axis) -> OwnedArray<A, <D as RemoveAxis>::Smaller>
        where A: LinalgScalar,
              D: RemoveAxis,
    {
        let n = self.shape()[axis.axis()];
        let mut sum = self.sum(axis);
        let one = libnum::one::<A>();
        let mut cnt = one;
        for _ in 1..n {
            cnt = cnt + one;
        }
        sum.idiv_scalar(&cnt);
        sum
    }

    /// Return `true` if the arrays' elementwise differences are all within
    /// the given absolute tolerance.<br>
    /// Return `false` otherwise, or if the shapes disagree.
    pub fn allclose<S2>(&self, rhs: &ArrayBase<S2, D>, tol: A) -> bool
        where A: Float,
              S2: Data<Elem=A>,
    {
        self.shape() == rhs.shape() &&
        self.iter().zip(rhs.iter()).all(|(x, y)| (*x - *y).abs() <= tol)
    }
}

impl<A, S> ArrayBase<S, Ix>
    where S: Data<Elem=A>,
{
    /// Compute the dot product of one-dimensional arrays.
    ///
    /// The dot product is a sum of the elementwise products (no conjugation
    /// of complex operands, and thus not their inner product).
    ///
    /// **Panics** if the arrays are not of the same length.
    pub fn dot<S2>(&self, rhs: &ArrayBase<S2, Ix>) -> A
        where S2: Data<Elem=A>,
              A: LinalgScalar,
    {
        self.dot_impl(rhs)
    }

    fn dot_generic<S2>(&self, rhs: &ArrayBase<S2, Ix>) -> A
        where S2: Data<Elem=A>,
              A: LinalgScalar,
    {
        assert_eq!(self.len(), rhs.len());
        if let Some(self_s) = self.as_slice() {
            if let Some(rhs_s) = rhs.as_slice() {
                return numeric_util::unrolled_dot(self_s, rhs_s);
            }
        }
        let mut sum = A::zero();
        for i in 0..self.len() {
            unsafe {
                sum = sum.clone() + self.uget(i).clone() * rhs.uget(i).clone();
            }
        }
        sum
    }

    #[cfg(not(feature="rblas"))]
    fn dot_impl<S2>(&self, rhs: &ArrayBase<S2, Ix>) -> A
        where S2: Data<Elem=A>,
              A: LinalgScalar,
    {
        self.dot_generic(rhs)
    }

    #[cfg(feature="rblas")]
    fn dot_impl<S2>(&self, rhs: &ArrayBase<S2, Ix>) -> A
        where S2: Data<Elem=A>,
              A: LinalgScalar,
    {
        use std::any::{Any, TypeId};
        use rblas::vector::ops::Dot;
        use linalg::AsBlasAny;

        // Read pointer to type `A` as type `B`.
        //
        // **Panics** if `A` and `B` are not the same type
        fn cast_as<A: Any + Copy, B: Any + Copy>(a: &A) -> B {
            assert_eq!(TypeId::of::<A>(), TypeId::of::<B>());
            unsafe {
                ::std::ptr::read(a as *const _ as *const B)
            }
        }
        // Use only if the vector is large enough to be worth it
        if self.len() >= 32 {
            assert_eq!(self.len(), rhs.len());
            if let Ok(self_v) = self.blas_view_as_type::<f32>() {
                if let Ok(rhs_v) = rhs.blas_view_as_type::<f32>() {
                    let f_ret = f32::dot(&self_v, &rhs_v);
                    return cast_as::<f32, A>(&f_ret);
                }
            }
            if let Ok(self_v) = self.blas_view_as_type::<f64>() {
                if let Ok(rhs_v) = rhs.blas_view_as_type::<f64>() {
                    let f_ret = f64::dot(&self_v, &rhs_v);
                    return cast_as::<f64, A>(&f_ret);
                }
            }
        }
        self.dot_generic(rhs)
    }
}


impl<A, S> ArrayBase<S, (Ix, Ix)>
    where S: Data<Elem=A>,
{
    /// Return an array view of row `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn row(&self, index: Ix) -> ArrayView<A, Ix>
    {
        self.subview(Axis(0), index)
    }

    /// Return a mutable array view of row `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn row_mut(&mut self, index: Ix) -> ArrayViewMut<A, Ix>
        where S: DataMut
    {
        self.subview_mut(Axis(0), index)
    }

    /// Return an array view of column `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn column(&self, index: Ix) -> ArrayView<A, Ix>
    {
        self.subview(Axis(1), index)
    }

    /// Return a mutable array view of column `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn column_mut(&mut self, index: Ix) -> ArrayViewMut<A, Ix>
        where S: DataMut
    {
        self.subview_mut(Axis(1), index)
    }

    /// Perform matrix multiplication of rectangular arrays `self` and `rhs`.
    ///
    /// The array shapes must agree in the way that
    /// if `self` is *M* × *N*, then `rhs` is *N* × *K*.
    ///
    /// Return a result array with shape *M* × *K*.
    ///
    /// **Panics** if shapes are incompatible.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [0., 1.]]);
    /// let b = arr2(&[[1., 2.],
    ///                [2., 3.]]);
    ///
    /// assert!(
    ///     a.mat_mul(&b) == arr2(&[[5., 8.],
    ///                             [2., 3.]])
    /// );
    /// ```
    ///
    pub fn mat_mul(&self, rhs: &ArrayBase<S, (Ix, Ix)>) -> OwnedArray<A, (Ix, Ix)>
        where A: LinalgScalar,
    {
        // NOTE: Matrix multiplication only defined for Copy types to
        // avoid trouble with panicking + and *, and destructors

        let ((m, a), (b, n)) = (self.dim, rhs.dim);
        let (self_columns, other_rows) = (a, b);
        assert!(self_columns == other_rows);

        // Avoid initializing the memory in vec -- set it during iteration
        // Panic safe because A: Copy
        let mut res_elems = Vec::<A>::with_capacity(m as usize * n as usize);
        unsafe {
            res_elems.set_len(m as usize * n as usize);
        }
        let mut i = 0;
        let mut j = 0;
        for rr in &mut res_elems {
            unsafe {
                *rr = (0..a).fold(libnum::zero::<A>(),
                    move |s, k| s + *self.uget((i, k)) * *rhs.uget((k, j))
                );
            }
            j += 1;
            if j == n {
                j = 0;
                i += 1;
            }
        }
        unsafe {
            ArrayBase::from_vec_dim_unchecked((m, n), res_elems)
        }
    }

    /// Perform the matrix multiplication of the rectangular array `self` and
    /// column vector `rhs`.
    ///
    /// The array shapes must agree in the way that
    /// if `self` is *M* × *N*, then `rhs` is *N*.
    ///
    /// Return a result array with shape *M*.
    ///
    /// **Panics** if shapes are incompatible.
    pub fn mat_mul_col(&self, rhs: &ArrayBase<S, Ix>) -> OwnedArray<A, Ix>
        where A: LinalgScalar,
    {
        let ((m, a), n) = (self.dim, rhs.dim);
        let (self_columns, other_rows) = (a, n);
        assert!(self_columns == other_rows);

        // Avoid initializing the memory in vec -- set it during iteration
        let mut res_elems = Vec::<A>::with_capacity(m as usize);
        unsafe {
            res_elems.set_len(m as usize);
        }
        for (i, rr) in enumerate(&mut res_elems) {
            unsafe {
                *rr = (0..a).fold(libnum::zero::<A>(),
                    move |s, k| s + *self.uget((i, k)) * *rhs.uget(k)
                );
            }
        }
        unsafe {
            ArrayBase::from_vec_dim_unchecked(m, res_elems)
        }
    }
}



// array OPERATORS

macro_rules! impl_binary_op_inherent(
    ($trt:ident, $mth:ident, $imethod:ident, $imth_scalar:ident, $doc:expr) => (
    /// Perform elementwise
    #[doc=$doc]
    /// between `self` and `rhs`,
    /// *in place*.
    ///
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting isn’t possible.
    pub fn $imethod <E: Dimension, S2> (&mut self, rhs: &ArrayBase<S2, E>)
        where A: Clone + $trt<A, Output=A>,
              S2: Data<Elem=A>,
    {
        self.zip_mut_with(rhs, |x, y| {
            *x = x.clone().$mth(y.clone());
        });
    }

    /// Perform elementwise
    #[doc=$doc]
    /// between `self` and the scalar `x`,
    /// *in place*.
    pub fn $imth_scalar (&mut self, x: &A)
        where A: Clone + $trt<A, Output=A>,
    {
        self.unordered_foreach_mut(move |elt| {
            *elt = elt.clone(). $mth (x.clone());
        });
    }
    );
);

/// *In-place* arithmetic operations.
impl<A, S, D> ArrayBase<S, D>
    where S: DataMut<Elem=A>,
          D: Dimension,
{


impl_binary_op_inherent!(Add, add, iadd, iadd_scalar, "addition");
impl_binary_op_inherent!(Sub, sub, isub, isub_scalar, "subtraction");
impl_binary_op_inherent!(Mul, mul, imul, imul_scalar, "multiplication");
impl_binary_op_inherent!(Div, div, idiv, idiv_scalar, "division");
impl_binary_op_inherent!(Rem, rem, irem, irem_scalar, "remainder");
impl_binary_op_inherent!(BitAnd, bitand, ibitand, ibitand_scalar, "bit and");
impl_binary_op_inherent!(BitOr, bitor, ibitor, ibitor_scalar, "bit or");
impl_binary_op_inherent!(BitXor, bitxor, ibitxor, ibitxor_scalar, "bit xor");
impl_binary_op_inherent!(Shl, shl, ishl, ishl_scalar, "left shift");
impl_binary_op_inherent!(Shr, shr, ishr, ishr_scalar, "right shift");

    /// Perform an elementwise negation of `self`, *in place*.
    pub fn ineg(&mut self)
        where A: Clone + Neg<Output=A>,
    {
        self.unordered_foreach_mut(|elt| {
            *elt = elt.clone().neg()
        });
    }

    /// Perform an elementwise unary not of `self`, *in place*.
    pub fn inot(&mut self)
        where A: Clone + Not<Output=A>,
    {
        self.unordered_foreach_mut(|elt| {
            *elt = elt.clone().not()
        });
    }

}

/// Elements that can be used as direct operands in arithmetic with arrays.
///
/// This trait ***does not*** limit which elements can be stored in an `ArrayBase`.
///
/// `Scalar` simply determines which types are applicable for direct operator
/// overloading, e.g. `B @ K` or `B @= K`  where `B` is a mutable array,
/// `K` a scalar, and `@` arbitrary arithmetic operator that the scalar supports.
///
/// Left hand side operands must instead be implemented one by one (it does not
/// involve the `Scalar` trait). Scalar left hand side operations: `K @ &A`
/// and `K @ B`, are implemented for the primitive numerical types and for
/// `Complex<f32>, Complex<f64>`.
///
/// Non-`Scalar` types can still participate in arithmetic as array elements in
/// in array-array operations.
pub trait Scalar { }
impl Scalar for bool { }
impl Scalar for i8 { }
impl Scalar for u8 { }
impl Scalar for i16 { }
impl Scalar for u16 { }
impl Scalar for i32 { }
impl Scalar for u32 { }
impl Scalar for i64 { }
impl Scalar for u64 { }
impl Scalar for f32 { }
impl Scalar for f64 { }
impl Scalar for Complex<f32> { }
impl Scalar for Complex<f64> { }

macro_rules! impl_binary_op(
    ($trt:ident, $mth:ident, $imth:ident, $imth_scalar:ident, $doc:expr) => (
/// Perform elementwise
#[doc=$doc]
/// between `self` and `rhs`,
/// and return the result (based on `self`).
///
/// `self` must be an `OwnedArray` or `RcArray`.
///
/// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
///
/// **Panics** if broadcasting isn’t possible.
impl<A, S, S2, D, E> $trt<ArrayBase<S2, E>> for ArrayBase<S, D>
    where A: Clone + $trt<A, Output=A>,
          S: DataOwned<Elem=A> + DataMut,
          S2: Data<Elem=A>,
          D: Dimension,
          E: Dimension,
{
    type Output = ArrayBase<S, D>;
    fn $mth(self, rhs: ArrayBase<S2, E>) -> ArrayBase<S, D>
    {
        self.$mth(&rhs)
    }
}

/// Perform elementwise
#[doc=$doc]
/// between `self` and reference `rhs`,
/// and return the result (based on `self`).
///
/// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
///
/// **Panics** if broadcasting isn’t possible.
impl<'a, A, S, S2, D, E> $trt<&'a ArrayBase<S2, E>> for ArrayBase<S, D>
    where A: Clone + $trt<A, Output=A>,
          S: DataMut<Elem=A>,
          S2: Data<Elem=A>,
          D: Dimension,
          E: Dimension,
{
    type Output = ArrayBase<S, D>;
    fn $mth (mut self, rhs: &ArrayBase<S2, E>) -> ArrayBase<S, D>
    {
        self.$imth(rhs);
        self
    }
}

/// Perform elementwise
#[doc=$doc]
/// between references `self` and `rhs`,
/// and return the result as a new `OwnedArray`.
///
/// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
///
/// **Panics** if broadcasting isn’t possible.
impl<'a, A, S, S2, D, E> $trt<&'a ArrayBase<S2, E>> for &'a ArrayBase<S, D>
    where A: Clone + $trt<A, Output=A>,
          S: Data<Elem=A>,
          S2: Data<Elem=A>,
          D: Dimension,
          E: Dimension,
{
    type Output = OwnedArray<A, D>;
    fn $mth (self, rhs: &'a ArrayBase<S2, E>) -> OwnedArray<A, D>
    {
        // FIXME: Can we co-broadcast arrays here? And how?
        self.to_owned().$mth(rhs)
    }
}

/// Perform elementwise
#[doc=$doc]
/// between `self` and the scalar `x`,
/// and return the result (based on `self`).
///
/// `self` must be an `OwnedArray` or `RcArray`.
impl<A, S, D, B> $trt<B> for ArrayBase<S, D>
    where A: Clone + $trt<B, Output=A>,
          S: DataOwned<Elem=A> + DataMut,
          D: Dimension,
          B: Clone + Scalar,
{
    type Output = ArrayBase<S, D>;
    fn $mth (mut self, x: B) -> ArrayBase<S, D>
    {
        self.unordered_foreach_mut(move |elt| {
            *elt = elt.clone().$mth(x.clone());
        });
        self
    }
}

/// Perform elementwise
#[doc=$doc]
/// between the reference `self` and the scalar `x`,
/// and return the result as a new `OwnedArray`.
impl<'a, A, S, D, B> $trt<B> for &'a ArrayBase<S, D>
    where A: Clone + $trt<B, Output=A>,
          S: Data<Elem=A>,
          D: Dimension,
          B: Clone + Scalar,
{
    type Output = OwnedArray<A, D>;
    fn $mth(self, x: B) -> OwnedArray<A, D>
    {
        self.to_owned().$mth(x)
    }
}
    );
);

macro_rules! impl_scalar_op {
    ($scalar:ty, $trt:ident, $mth:ident, $doc:expr) => (
// these have no doc -- they are not visible in rustdoc
// Perform elementwise
// between the scalar `self` and array `rhs`,
// and return the result (based on `self`).
impl<S, D> $trt<ArrayBase<S, D>> for $scalar
    where S: DataMut<Elem=$scalar>,
          D: Dimension,
{
    type Output = ArrayBase<S, D>;
    fn $mth (self, mut rhs: ArrayBase<S, D>) -> ArrayBase<S, D>
    {
        rhs.unordered_foreach_mut(move |elt| {
            *elt = self.$mth(*elt);
        });
        rhs
    }
}

// Perform elementwise
// between the scalar `self` and array `rhs`,
// and return the result as a new `OwnedArray`.
impl<'a, S, D> $trt<&'a ArrayBase<S, D>> for $scalar
    where S: Data<Elem=$scalar>,
          D: Dimension,
{
    type Output = OwnedArray<$scalar, D>;
    fn $mth (self, rhs: &ArrayBase<S, D>) -> OwnedArray<$scalar, D>
    {
        self.$mth(rhs.to_owned())
    }
}
    );
}


mod arithmetic_ops {
    use super::*;
    use std::ops::*;
    use libnum::Complex;

    impl_binary_op!(Add, add, iadd, iadd_scalar, "addition");
    impl_binary_op!(Sub, sub, isub, isub_scalar, "subtraction");
    impl_binary_op!(Mul, mul, imul, imul_scalar, "multiplication");
    impl_binary_op!(Div, div, idiv, idiv_scalar, "division");
    impl_binary_op!(Rem, rem, irem, irem_scalar, "remainder");
    impl_binary_op!(BitAnd, bitand, ibitand, ibitand_scalar, "bit and");
    impl_binary_op!(BitOr, bitor, ibitor, ibitor_scalar, "bit or");
    impl_binary_op!(BitXor, bitxor, ibitxor, ibitxor_scalar, "bit xor");
    impl_binary_op!(Shl, shl, ishl, ishl_scalar, "left shift");
    impl_binary_op!(Shr, shr, ishr, ishr_scalar, "right shift");

    macro_rules! all_scalar_ops {
        ($int_scalar:ty) => (
            impl_scalar_op!($int_scalar, Add, add, "addition");
            impl_scalar_op!($int_scalar, Sub, sub, "subtraction");
            impl_scalar_op!($int_scalar, Mul, mul, "multiplication");
            impl_scalar_op!($int_scalar, Div, div, "division");
            impl_scalar_op!($int_scalar, Rem, rem, "remainder");
            impl_scalar_op!($int_scalar, BitAnd, bitand, "bit and");
            impl_scalar_op!($int_scalar, BitOr, bitor, "bit or");
            impl_scalar_op!($int_scalar, BitXor, bitxor, "bit xor");
            impl_scalar_op!($int_scalar, Shl, shl, "left shift");
            impl_scalar_op!($int_scalar, Shr, shr, "right shift");
        );
    }
    all_scalar_ops!(i8);
    all_scalar_ops!(u8);
    all_scalar_ops!(i16);
    all_scalar_ops!(u16);
    all_scalar_ops!(i32);
    all_scalar_ops!(u32);
    all_scalar_ops!(i64);
    all_scalar_ops!(u64);

    impl_scalar_op!(bool, BitAnd, bitand, "bit and");
    impl_scalar_op!(bool, BitOr, bitor, "bit or");
    impl_scalar_op!(bool, BitXor, bitxor, "bit xor");

    impl_scalar_op!(f32, Add, add, "addition");
    impl_scalar_op!(f32, Sub, sub, "subtraction");
    impl_scalar_op!(f32, Mul, mul, "multiplication");
    impl_scalar_op!(f32, Div, div, "division");
    impl_scalar_op!(f32, Rem, rem, "remainder");

    impl_scalar_op!(f64, Add, add, "addition");
    impl_scalar_op!(f64, Sub, sub, "subtraction");
    impl_scalar_op!(f64, Mul, mul, "multiplication");
    impl_scalar_op!(f64, Div, div, "division");
    impl_scalar_op!(f64, Rem, rem, "remainder");

    impl_scalar_op!(Complex<f32>, Add, add, "addition");
    impl_scalar_op!(Complex<f32>, Sub, sub, "subtraction");
    impl_scalar_op!(Complex<f32>, Mul, mul, "multiplication");
    impl_scalar_op!(Complex<f32>, Div, div, "division");

    impl_scalar_op!(Complex<f64>, Add, add, "addition");
    impl_scalar_op!(Complex<f64>, Sub, sub, "subtraction");
    impl_scalar_op!(Complex<f64>, Mul, mul, "multiplication");
    impl_scalar_op!(Complex<f64>, Div, div, "division");

    impl<A, S, D> Neg for ArrayBase<S, D>
        where A: Clone + Neg<Output=A>,
              S: DataMut<Elem=A>,
              D: Dimension
    {
        type Output = Self;
        /// Perform an elementwise negation of `self` and return the result.
        fn neg(mut self) -> Self {
            self.ineg();
            self
        }
    }

    impl<A, S, D> Not for ArrayBase<S, D>
        where A: Clone + Not<Output=A>,
              S: DataMut<Elem=A>,
              D: Dimension
    {
        type Output = Self;
        /// Perform an elementwise unary not of `self` and return the result.
        fn not(mut self) -> Self {
            self.inot();
            self
        }
    }
}

#[cfg(feature = "assign_ops")]
mod assign_ops {
    use super::*;

    macro_rules! impl_assign_op {
        ($trt:ident, $method:ident, $doc:expr) => {
    use std::ops::$trt;

    #[doc=$doc]
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting isn’t possible.
    ///
    /// **Requires crate feature `"assign_ops"`**
    impl<'a, A, S, S2, D, E> $trt<&'a ArrayBase<S2, E>> for ArrayBase<S, D>
        where A: Clone + $trt<A>,
              S: DataMut<Elem=A>,
              S2: Data<Elem=A>,
              D: Dimension,
              E: Dimension,
    {
        fn $method(&mut self, rhs: &ArrayBase<S2, E>) {
            self.zip_mut_with(rhs, |x, y| {
                x.$method(y.clone());
            });
        }
    }

    #[doc=$doc]
    /// **Requires crate feature `"assign_ops"`**
    impl<A, S, D, B> $trt<B> for ArrayBase<S, D>
        where A: $trt<B>,
              S: DataMut<Elem=A>,
              D: Dimension,
              B: Clone + Scalar,
    {
        fn $method(&mut self, rhs: B) {
            self.unordered_foreach_mut(move |elt| {
                elt.$method(rhs.clone());
            });
        }
    }

        };
    }

    impl_assign_op!(AddAssign, add_assign,
                    "Perform `self += rhs` as elementwise addition (in place).\n");
    impl_assign_op!(SubAssign, sub_assign,
                    "Perform `self -= rhs` as elementwise subtraction (in place).\n");
    impl_assign_op!(MulAssign, mul_assign,
                    "Perform `self *= rhs` as elementwise multiplication (in place).\n");
    impl_assign_op!(DivAssign, div_assign,
                    "Perform `self /= rhs` as elementwise division (in place).\n");
    impl_assign_op!(RemAssign, rem_assign,
                    "Perform `self %= rhs` as elementwise remainder (in place).\n");
    impl_assign_op!(BitAndAssign, bitand_assign,
                    "Perform `self &= rhs` as elementwise bit and (in place).\n");
    impl_assign_op!(BitOrAssign, bitor_assign,
                    "Perform `self |= rhs` as elementwise bit or (in place).\n");
    impl_assign_op!(BitXorAssign, bitxor_assign,
                    "Perform `self ^= rhs` as elementwise bit xor (in place).\n");
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

