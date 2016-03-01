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
//!   `ArrayView::from_slice` and `ArrayViewMut::from_slice`.
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
//! The following crate feature flags are available. The are configured in your
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

use std::any::Any;
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

use dimension::stride_offset;

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
        RemoveAxis,
        Data,
        DataMut,
        DataOwned,
        DataShared,
    };
    /// Wrapper type for private methods
    #[derive(Copy, Clone, Debug)]
    pub struct Priv<T>(pub T);
}

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
/// The trait [`ScalarOperand`](trait.ScalarOperand.html) marks types that can be used in arithmetic
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

mod impl_clone;

mod impl_constructors;

mod impl_methods;

/// Private Methods
impl<A, S, D> ArrayBase<S, D>
    where S: Data<Elem=A>, D: Dimension
{
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

macro_rules! impl_binary_op_inplace(
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


impl_binary_op_inplace!(Add, add, iadd, iadd_scalar, "addition");
impl_binary_op_inplace!(Sub, sub, isub, isub_scalar, "subtraction");
impl_binary_op_inplace!(Mul, mul, imul, imul_scalar, "multiplication");
impl_binary_op_inplace!(Div, div, idiv, idiv_scalar, "division");
impl_binary_op_inplace!(Rem, rem, irem, irem_scalar, "remainder");
impl_binary_op_inplace!(BitAnd, bitand, ibitand, ibitand_scalar, "bit and");
impl_binary_op_inplace!(BitOr, bitor, ibitor, ibitor_scalar, "bit or");
impl_binary_op_inplace!(BitXor, bitxor, ibitxor, ibitxor_scalar, "bit xor");
impl_binary_op_inplace!(Shl, shl, ishl, ishl_scalar, "left shift");
impl_binary_op_inplace!(Shr, shr, ishr, ishr_scalar, "right shift");

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
/// For example, `f64` is a `ScalarOperand` which means that for an array `a`,
/// arithmetic like `a + 1.0`, and, `a * 2.`, and `a += 3.` are allowed.
///
/// In the description below, let `A` be an array or array view,
/// let `B` be an array with owned data,
/// and let `C` be an array with mutable data.
///
/// `ScalarOperand` determines for which scalars `K` operations `&A @ K`, and `B @ K`,
/// and `C @= K` are defined, as **right hand side** operands, for applicable
/// arithmetic operators (denoted `@`).
///
/// **Left hand side** scalar operands are implemented differently
/// (one `impl` per concrete scalar type); they are
/// implemented for the default `ScalarOperand` types, allowing
/// operations `K @ &A`, and `K @ B`.
///
/// This trait ***does not*** limit which elements can be stored in an array in general.
/// Non-`ScalarOperand` types can still participate in arithmetic as array elements in
/// in array-array operations.
pub trait ScalarOperand : Any + Clone { }
impl ScalarOperand for bool { }
impl ScalarOperand for i8 { }
impl ScalarOperand for u8 { }
impl ScalarOperand for i16 { }
impl ScalarOperand for u16 { }
impl ScalarOperand for i32 { }
impl ScalarOperand for u32 { }
impl ScalarOperand for i64 { }
impl ScalarOperand for u64 { }
impl ScalarOperand for f32 { }
impl ScalarOperand for f64 { }
impl ScalarOperand for Complex<f32> { }
impl ScalarOperand for Complex<f64> { }

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
          B: ScalarOperand,
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
          B: ScalarOperand,
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
    impl<A, S, D> $trt<A> for ArrayBase<S, D>
        where A: ScalarOperand + $trt<A>,
              S: DataMut<Elem=A>,
              D: Dimension,
    {
        fn $method(&mut self, rhs: A) {
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
    impl_assign_op!(ShlAssign, shl_assign,
                    "Perform `self <<= rhs` as elementwise left shift (in place).\n");
    impl_assign_op!(ShrAssign, shr_assign,
                    "Perform `self >>= rhs` as elementwise right shift (in place).\n");
}

/// # Methods for Array Views
///
/// Methods for read-only array views `ArrayView<'a, A, D>`
impl<'a, A> ArrayBase<ViewRepr<&'a A>, Ix> {
    /// Create a one-dimensional read-only array view of the data in `xs`.
    #[inline]
    pub fn from_slice(xs: &'a [A]) -> Self {
        ArrayView {
            data: ViewRepr::new(),
            ptr: xs.as_ptr() as *mut A,
            dim: xs.len(),
            strides: 1,
        }
    }
}

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

    /// Create a read-only array view borrowing its data from a slice.
    ///
    /// Checks whether `dim` and `strides` are compatible with the slice's
    /// length, returning an `Err` if not compatible.
    ///
    /// ```
    /// use ndarray::ArrayView;
    /// use ndarray::arr3;
    ///
    /// let s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    /// let a = ArrayView::from_slice_dim_stride((2, 3, 2),
    ///                                          (1, 4, 2),
    ///                                          &s).unwrap();
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
    pub fn from_slice_dim_stride(dim: D, strides: D, xs: &'a [A])
        -> Result<Self, StrideError>
    {
        dimension::can_index_slice(xs, &dim, &strides).map(|_| {
            unsafe {
                Self::new_(xs.as_ptr(), dim, strides)
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
    #[doc(hidden)] // not official
    #[cfg_attr(has_deprecated, deprecated(note="This method will be replaced."))]
    pub fn into_outer_iter(self) -> OuterIter<'a, A, D::Smaller>
        where D: RemoveAxis,
    {
        iterators::new_outer_iter(self)
    }

    /// Split the array along `axis` and return one view strictly before the
    /// split and one view after the split.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    pub fn split_at(self, axis: Axis, index: Ix)
        -> (Self, Self)
    {
        // NOTE: Keep this in sync with the ArrayViewMut version
        let axis = axis.axis();
        assert!(index <= self.shape()[axis]);
        let left_ptr = self.ptr;
        let right_ptr = if index == self.shape()[axis] {
            self.ptr
        } else {
            let offset = stride_offset(index, self.strides.slice()[axis]);
            unsafe {
                self.ptr.offset(offset)
            }
        };

        let mut dim_left = self.dim.clone();
        dim_left.slice_mut()[axis] = index;
        let left = unsafe {
            Self::new_(left_ptr, dim_left, self.strides.clone())
        };

        let mut dim_right = self.dim;
        let right_len  = dim_right.slice()[axis] - index;
        dim_right.slice_mut()[axis] = right_len;
        let right = unsafe {
            Self::new_(right_ptr, dim_right, self.strides)
        };

        (left, right)
    }

}

/// Methods for read-write array views `ArrayViewMut<'a, A, D>`
impl<'a, A> ArrayBase<ViewRepr<&'a mut A>, Ix> {
    /// Create a one-dimensional read-write array view of the data in `xs`.
    #[inline]
    pub fn from_slice(xs: &'a mut [A]) -> Self {
        ArrayViewMut {
            data: ViewRepr::new(),
            ptr: xs.as_mut_ptr(),
            dim: xs.len(),
            strides: 1,
        }
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

    /// Create a read-write array view borrowing its data from a slice.
    ///
    /// Checks whether `dim` and `strides` are compatible with the slice's
    /// length, returning an `Err` if not compatible.
    ///
    /// ```
    /// use ndarray::ArrayViewMut;
    /// use ndarray::arr3;
    ///
    /// let mut s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    /// let mut a = ArrayViewMut::from_slice_dim_stride((2, 3, 2),
    ///                                                 (1, 4, 2),
    ///                                                 &mut s).unwrap();
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
    pub fn from_slice_dim_stride(dim: D, strides: D, xs: &'a mut [A])
        -> Result<Self, StrideError>
    {
        dimension::can_index_slice(xs, &dim, &strides).map(|_| {
            unsafe {
                Self::new_(xs.as_mut_ptr(), dim, strides)
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

    /// Return an outer iterator for this view.
    #[doc(hidden)] // not official
    #[cfg_attr(has_deprecated, deprecated(note="This method will be replaced."))]
    pub fn into_outer_iter(self) -> OuterIterMut<'a, A, D::Smaller>
        where D: RemoveAxis,
    {
        iterators::new_outer_iter_mut(self)
    }

    /// Split the array along `axis` and return one mutable view strictly
    /// before the split and one mutable view after the split.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    pub fn split_at(self, axis: Axis, index: Ix)
        -> (Self, Self)
    {
        // NOTE: Keep this in sync with the ArrayView version
        let axis = axis.axis();
        assert!(index <= self.shape()[axis]);
        let left_ptr = self.ptr;
        let right_ptr = if index == self.shape()[axis] {
            self.ptr
        }
        else {
            let offset = stride_offset(index, self.strides.slice()[axis]);
            unsafe {
                self.ptr.offset(offset)
            }
        };

        let mut dim_left = self.dim.clone();
        dim_left.slice_mut()[axis] = index;
        let left = unsafe {
            Self::new_(left_ptr, dim_left, self.strides.clone())
        };

        let mut dim_right = self.dim;
        let right_len  = dim_right.slice()[axis] - index;
        dim_right.slice_mut()[axis] = right_len;
        let right = unsafe {
            Self::new_(right_ptr, dim_right, self.strides)
        };

        (left, right)
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

