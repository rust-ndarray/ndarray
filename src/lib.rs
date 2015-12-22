#![crate_name="ndarray"]
#![cfg_attr(has_deprecated, feature(deprecated))]
#![doc(html_root_url = "http://bluss.github.io/rust-ndarray/doc/")]

//! The `ndarray` crate provides an N-dimensional container similar to numpy’s
//! ndarray.
//!
//! - [`ArrayBase`](struct.ArrayBase.html):
//!   The N-dimensional array type itself.
//! - [`Array`](type.Array.html):
//!   An array where the data is shared and copy on write, it
//!   can act as both an owner of the data as well as a lightweight view.
//! - [`OwnedArray`](type.OwnedArray.html):
//!   An array where the data is owned uniquely.
//! - [`ArrayView`](type.ArrayView.html), [`ArrayViewMut`](type.ArrayViewMut.html):
//!   Lightweight array views.
//!
//! ## Highlights
//!
//! - Generic N-dimensional array
//! - Slicing, also with arbitrary step size, and negative indices to mean
//!   elements from the end of the axis.
//! - There is both an easy to use copy on write array (`Array`),
//!   or a regular uniquely owned array (`OwnedArray`), and both can use
//!   read-only and read-write array views.
//! - Iteration and most operations are very efficient on contiguous c-order arrays
//!   (the default layout, without any transposition or discontiguous subslicing),
//!   and on arrays where the lowest dimension is contiguous (contiguous block
//!   slicing).
//! - Array views can be used to slice and mutate any `[T]` data.
//!
//! ## Status and Lookout
//!
//! - Still iterating on the API
//! - Performance status:
//!   + Arithmetic involving contiguous c-order arrays and contiguous lowest
//!     dimension arrays optimizes very well.
//!   + `.fold()` and `.zip_mut_with()` are the most efficient ways to
//!     perform single traversal and lock step traversal respectively.
//!   + Transposed arrays where the lowest dimension is not c-contiguous
//!     is still a pain point.
//! - There is experimental bridging to the linear algebra package `rblas`.
//!
//! ## Crate Feature Flags
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

extern crate itertools as it;
#[cfg(not(nocomplex))]
extern crate num as libnum;

use libnum::Float;

use std::cmp;
use std::mem;
use std::ops::{Add, Sub, Mul, Div, Rem, Neg, Not, Shr, Shl,
    BitAnd,
    BitOr,
    BitXor,
};
use std::rc::Rc;
use std::slice::{self, Iter, IterMut};

use it::ZipSlices;

pub use dimension::{Dimension, RemoveAxis};
pub use indexes::Indexes;
pub use shape_error::ShapeError;
pub use si::{Si, S};

use dimension::stride_offset;
use iterators::Baseiter;
pub use iterators::{
    InnerIter,
    InnerIterMut,
};

#[allow(deprecated)]
use linalg::{Field, Ring};

pub mod linalg;
mod arraytraits;
#[cfg(feature = "serde")]
mod arrayserialize;
mod arrayformat;
#[cfg(feature = "rblas")]
pub mod blas;
mod dimension;
mod indexes;
mod iterators;
mod si;
mod shape_error;

// NOTE: In theory, the whole library should compile
// and pass tests even if you change Ix and Ixs.
/// Array index type
pub type Ix = u32;
/// Array index type (signed)
pub type Ixs = i32;

/// An *N*-dimensional array.
///
/// The array is a general container of elements. It can be of numerical use
/// too, supporting all mathematical operators by applying them elementwise.  It
/// cannot grow or shrink, but can be sliced into views of parts of its data.
///
/// The `ArrayBase<S, D>` is parameterized by:
///
/// - `S` for the data storage
/// - `D` for the number of dimensions
///
/// Type aliases [`Array`], [`OwnedArray`], [`ArrayView`], and [`ArrayViewMut`] refer
/// to `ArrayBase` with different types for the data storage.
///
/// [`Array`]: type.Array.html
/// [`OwnedArray`]: type.OwnedArray.html
/// [`ArrayView`]: type.ArrayView.html
/// [`ArrayViewMut`]: type.ArrayViewMut.html
///
/// ## `Array`
///
/// `Array<A, D>` is a an array with reference counted data and copy-on-write
/// mutability.
///
/// The `Array` is both a view and a shared owner of its data. Some methods,
/// for example [`slice()`](#method.slice), merely change the view of the data,
/// while methods like [`iadd()`](#method.iadd) allow mutating the element
/// values.
///
/// Calling a method for mutating elements, for example
/// [`get_mut()`](#method.get_mut), [`iadd()`](#method.iadd) or
/// [`iter_mut()`](#method.iter_mut) will break sharing and require a clone of
/// the data (if it is not uniquely held).
///
/// ## Method Conventions
///
/// Methods mutating the view or array elements in place use an *i* prefix,
/// for example `slice` vs. `islice` and `add` vs `iadd`.
///
/// Note that all `ArrayBase` variants can change their view (slicing) of the
/// data freely, even when the data can’t be mutated.
///
/// ## Indexing
///
/// Array indexes are represented by the types `Ix` and `Ixs`
/// (signed). ***Note: A future version will switch from `u32` to `usize`.***
///
/// ## Slicing
///
/// You can use slicing to create a view of a subset of the data in
/// the array. Slicing methods include `.slice()`, `.islice()`,
/// `.slice_mut()`.
///
/// The dimensionality of the array determines the number of *axes*, for example
/// a 2D array has two axes. These are listed in “big endian” order, so that
/// the greatest dimension is listed first, the lowest dimension with the most
/// rapidly varying index is the last.
/// For the 2D array this means that indices are `(row, column)`, and the order of
/// the elements is *(0, 0), (0, 1), (0, 2), ... (1, 0), (1, 1), (1, 2) ...* etc.
///
/// The slicing specification is passed as a function argument as a fixed size
/// array with elements of type [`Si`] with fields `Si(begin, end, stride)`,
/// where the values are signed integers, and `end` is an `Option<Ixs>`.
/// The constant [`S`] is a shorthand for the full range of an axis.
/// For example, if the array has two axes, the slice argument is passed as
/// type `&[Si; 2]`.
///
/// The macro [`s![]`](macro.s!.html) is however a much more convenient way to
/// specify the slicing argument, so it will be used in all examples.
///
/// [`Si`]: struct.Si.html
/// [`S`]: constant.S.html
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
/// use ndarray::{arr3, aview2};
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
/// let sub_0 = a.subview(0, 0);
/// let sub_1 = a.subview(0, 1);
///
/// assert_eq!(sub_0, aview2(&[[ 1,  2,  3],
///                            [ 4,  5,  6]]));
/// assert_eq!(sub_1, aview2(&[[ 7,  8,  9],
///                            [10, 11, 12]]));
/// assert_eq!(sub_0.shape(), &[2, 3]);
///
/// // This is the subview picking only axis 2, column 0
/// let sub_col = a.subview(2, 0);
///
/// assert_eq!(sub_col, aview2(&[[ 1,  4],
///                              [ 7, 10]]));
/// ```
///
/// `.isubview()` modifies the view in the same way as `subview()`, but
/// since it is *in place*, it cannot remove the collapsed axis. It becomes
/// an axis of length 1.
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
pub struct ArrayBase<S, D> where S: Data {
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
pub unsafe trait Data {
    type Elem;
    fn slice(&self) -> &[Self::Elem];
}

/// Array’s writable inner representation.
pub unsafe trait DataMut : Data {
    fn slice_mut(&mut self) -> &mut [Self::Elem];
    fn ensure_unique<D>(&mut ArrayBase<Self, D>)
        where Self: Sized, D: Dimension
    {
    }
}

/// Clone an Array’s storage.
pub unsafe trait DataClone : Data {
    /// Unsafe because, `ptr` must point inside the current storage.
    unsafe fn clone_with_ptr(&self, ptr: *mut Self::Elem) -> (Self, *mut Self::Elem);
}

unsafe impl<A> Data for Rc<Vec<A>> {
    type Elem = A;
    fn slice(&self) -> &[A] { self }
}

// NOTE: Copy on write
unsafe impl<A> DataMut for Rc<Vec<A>> where A: Clone {
    fn slice_mut(&mut self) -> &mut [A] { &mut Rc::make_mut(self)[..] }

    fn ensure_unique<D>(self_: &mut ArrayBase<Self, D>)
        where Self: Sized, D: Dimension
    {
        if Rc::get_mut(&mut self_.data).is_some() {
            return
        }
        if self_.dim.size() <= self_.data.len() / 2 {
            unsafe {
                *self_ = Array::from_vec_dim(self_.dim.clone(),
                                            self_.iter().map(|x| x.clone()).collect());
            }
            return;
        }
        let our_off = (self_.ptr as isize - self_.data.as_ptr() as isize)
            / mem::size_of::<A>() as isize;
        let rvec = Rc::make_mut(&mut self_.data);
        unsafe {
            self_.ptr = rvec.as_mut_ptr().offset(our_off);
        }
    }
}

unsafe impl<A> DataClone for Rc<Vec<A>> {
    unsafe fn clone_with_ptr(&self, ptr: *mut Self::Elem)
        -> (Self, *mut Self::Elem)
    {
        // pointer is preserved
        (self.clone(), ptr)
    }
}

unsafe impl<A> Data for Vec<A> {
    type Elem = A;
    fn slice(&self) -> &[A] { self }
}

unsafe impl<A> DataMut for Vec<A> {
    fn slice_mut(&mut self) -> &mut [A] { self }
}

unsafe impl<A> DataClone for Vec<A> where A: Clone {
    unsafe fn clone_with_ptr(&self, ptr: *mut Self::Elem)
        -> (Self, *mut Self::Elem)
    {
        let mut u = self.clone();
        let our_off = (self.as_ptr() as isize - ptr as isize)
            / mem::size_of::<A>() as isize;
        let new_ptr = u.as_mut_ptr().offset(our_off);
        (u, new_ptr)
    }
}

unsafe impl<'a, A> Data for &'a [A] {
    type Elem = A;
    fn slice(&self) -> &[A] { self }
}

unsafe impl<'a, A> DataClone for &'a [A] {
    unsafe fn clone_with_ptr(&self, ptr: *mut Self::Elem)
        -> (Self, *mut Self::Elem)
    {
        (*self, ptr)
    }
}

unsafe impl<'a, A> Data for &'a mut [A] {
    type Elem = A;
    fn slice(&self) -> &[A] { self }
}

unsafe impl<'a, A> DataMut for &'a mut [A] {
    fn slice_mut(&mut self) -> &mut [A] { self }
}

/// Array representation that is a unique or shared owner of its data.
pub unsafe trait DataOwned : Data {
    fn new(elements: Vec<Self::Elem>) -> Self;
    fn into_shared(self) -> Rc<Vec<Self::Elem>>;
}

/// Array representation that is a lightweight view.
pub unsafe trait DataShared : Clone + DataClone { }

unsafe impl<A> DataShared for Rc<Vec<A>> { }
unsafe impl<'a, A> DataShared for &'a [A] { }

unsafe impl<A> DataOwned for Vec<A> {
    fn new(elements: Vec<A>) -> Self { elements }
    fn into_shared(self) -> Rc<Vec<A>> { Rc::new(self) }
}

unsafe impl<A> DataOwned for Rc<Vec<A>> {
    fn new(elements: Vec<A>) -> Self { Rc::new(elements) }
    fn into_shared(self) -> Rc<Vec<A>> { self }
}


/// Array where the data is reference counted and copy on write, it
/// can act as both an owner as the data as well as a lightweight view.
pub type Array<A, D> = ArrayBase<Rc<Vec<A>>, D>;

/// Array where the data is owned uniquely.
pub type OwnedArray<A, D> = ArrayBase<Vec<A>, D>;

/// A lightweight array view.
pub type ArrayView<'a, A, D> = ArrayBase<&'a [A], D>;
/// A lightweight read-write array view.
pub type ArrayViewMut<'a, A, D> = ArrayBase<&'a mut [A], D>;

impl<S: DataClone, D: Clone> Clone for ArrayBase<S, D>
{
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

impl<S: DataClone + Copy, D: Copy> Copy for ArrayBase<S, D> { }

/// Constructor methods for single dimensional `ArrayBase`.
impl<S> ArrayBase<S, Ix>
    where S: DataOwned,
{
    /// Create a one-dimensional array from a vector (no allocation needed).
    pub fn from_vec(v: Vec<S::Elem>) -> ArrayBase<S, Ix> {
        unsafe {
            Self::from_vec_dim(v.len() as Ix, v)
        }
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
              usize: it::misc::ToFloat<F>,
    {
        Self::from_iter(it::linspace(start, end, n))
    }

    /// Create a one-dimensional array from interval `[start, end)`
    #[cfg_attr(has_deprecated, deprecated(note="use ArrayBase::linspace() instead"))]
    pub fn range(start: f32, end: f32) -> ArrayBase<S, Ix>
        where S: Data<Elem=f32>,
    {
        let n = (end - start) as usize;
        let span = if n > 0 { (n - 1) as f32 } else { 0. };
        Self::linspace(start, start + span, n)
    }
}

/// Constructor methods for `ArrayBase`.
impl<S, A, D> ArrayBase<S, D>
    where S: DataOwned<Elem=A>,
          D: Dimension,
{
    /// Construct an array with copies of `elem`, dimension `dim`.
    ///
    /// ```
    /// use ndarray::Array;
    /// use ndarray::arr3;
    ///
    /// let a = Array::from_elem((2, 2, 2), 1.);
    ///
    /// assert!(
    ///     a == arr3(&[[[1., 1.],
    ///                  [1., 1.]],
    ///                 [[1., 1.],
    ///                  [1., 1.]]])
    /// );
    /// ```
    pub fn from_elem(dim: D, elem: A) -> ArrayBase<S, D> where A: Clone
    {
        let v = vec![elem; dim.size()];
        unsafe {
            Self::from_vec_dim(dim, v)
        }
    }

    /// Construct an array with zeros, dimension `dim`.
    pub fn zeros(dim: D) -> ArrayBase<S, D> where A: Clone + libnum::Zero
    {
        Self::from_elem(dim, libnum::zero())
    }

    /// Construct an array with default values, dimension `dim`.
    pub fn default(dim: D) -> ArrayBase<S, D>
        where A: Default
    {
        let v = (0..dim.size()).map(|_| A::default()).collect();
        unsafe {
            Self::from_vec_dim(dim, v)
        }
    }

    /// Create an array from a vector (with no allocation needed).
    ///
    /// Unsafe because dimension is unchecked, and must be correct.
    pub unsafe fn from_vec_dim(dim: D, mut v: Vec<A>) -> ArrayBase<S, D>
    {
        debug_assert!(dim.size() == v.len());
        ArrayBase {
            ptr: v.as_mut_ptr(),
            data: DataOwned::new(v),
            strides: dim.default_strides(),
            dim: dim
        }
    }
}

impl<'a, A, D> ArrayView<'a, A, D>
    where D: Dimension,
{
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
            inner:
            if let Some(slc) = self.into_slice() {
                ElementsRepr::Slice(slc.iter())
            } else {
                ElementsRepr::Counted(self.into_elements_base())
            }
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

}

impl<'a, A, D> ArrayViewMut<'a, A, D>
    where D: Dimension,
{
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

}

impl<A, S, D> ArrayBase<S, D> where S: Data<Elem=A>, D: Dimension
{
    /// Return the total number of elements in the Array.
    pub fn len(&self) -> usize
    {
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

    /// Return a read-only view of the array
    pub fn view(&self) -> ArrayView<A, D> {
        debug_assert!(self.pointer_is_inbounds());
        ArrayView {
            ptr: self.ptr,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: self.raw_data(),
        }
    }

    /// Return a read-write view of the array
    pub fn view_mut(&mut self) -> ArrayViewMut<A, D>
        where S: DataMut,
    {
        self.ensure_unique();
        ArrayViewMut {
            ptr: self.ptr,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            data: self.data.slice_mut(),
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
            ArrayBase::from_vec_dim(self.dim.clone(), data)
        }
    }

    /// Return a shared ownership (copy on write) array.
    pub fn to_shared(&self) -> Array<A, D>
        where A: Clone
    {
        // FIXME: Avoid copying if it’s already an Array.
        self.to_owned().into_shared()
    }

    /// Turn the array into a shared ownership (copy on write) array,
    /// without any copying.
    pub fn into_shared(self) -> Array<A, D>
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

    /// Return an iterator of references to the elements of the array.
    ///
    /// Iterator element type is `(D, &A)`.
    pub fn indexed_iter(&self) -> Indexed<A, D> {
        Indexed(self.view().into_elements_base())
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
    ///
    /// [`D::SliceArg`] is typically a fixed size array of `Si`, with one
    /// element per axis.
    ///
    /// [`D::SliceArg`]: trait.Dimension.html#associatedtype.SliceArg
    ///
    /// **Panics** if an index is out of bounds or stride is zero.<br>
    /// (**Panics** if `D` is `Vec` and `indexes` does not match the number of array axes.)
    pub fn slice(&self, indexes: &D::SliceArg) -> Self
        where S: DataShared
    {
        let mut arr = self.clone();
        arr.islice(indexes);
        arr
    }

    /// Slice the array’s view in place.
    ///
    /// [`D::SliceArg`] is typically a fixed size array of `Si`, with one
    /// element per axis.
    ///
    /// [`D::SliceArg`]: trait.Dimension.html#associatedtype.SliceArg
    ///
    /// **Panics** if an index is out of bounds or stride is zero.<br>
    /// (**Panics** if `D` is `Vec` and `indexes` does not match the number of array axes.)
    pub fn islice(&mut self, indexes: &D::SliceArg)
    {
        let offset = Dimension::do_slices(&mut self.dim, &mut self.strides, indexes);
        unsafe {
            self.ptr = self.ptr.offset(offset);
        }
    }

    /// Return an iterator over a sliced view.
    ///
    /// [`D::SliceArg`] is typically a fixed size array of `Si`, with one
    /// element per axis.
    ///
    /// [`D::SliceArg`]: trait.Dimension.html#associatedtype.SliceArg
    ///
    /// **Panics** if an index is out of bounds or stride is zero.<br>
    /// (**Panics** if `D` is `Vec` and `indexes` does not match the number of array axes.)
    pub fn slice_iter(&self, indexes: &D::SliceArg) -> Elements<A, D>
    {
        let mut it = self.view();
        it.islice(indexes);
        it.into_iter_()
    }

    /// Return a sliced read-write view of the array.
    ///
    /// [`D::SliceArg`] is typically a fixed size array of `Si`, with one
    /// element per axis.
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

    /// ***Deprecated: use `.slice_mut()`***
    #[cfg_attr(has_deprecated, deprecated(note="use .slice_mut() instead"))]
    pub fn slice_iter_mut(&mut self, indexes: &D::SliceArg) -> ElementsMut<A, D>
        where S: DataMut,
    {
        self.slice_mut(indexes).into_iter()
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
    ///     a[(0, 1)] == 2.
    /// );
    /// ```
    pub fn get(&self, index: D) -> Option<&A> {
        let ptr = self.ptr;
        self.dim.stride_offset_checked(&self.strides, &index)
            .map(move |offset| unsafe {
                &*ptr.offset(offset)
            })
    }

    /// ***Deprecated: use .get(i)***
    #[cfg_attr(has_deprecated, deprecated(note="use .get() instead"))]
    pub fn at(&self, index: D) -> Option<&A> {
        self.get(index)
    }

    /// Return a mutable reference to the element at `index`, or return `None`
    /// if the index is out of bounds.
    pub fn get_mut(&mut self, index: D) -> Option<&mut A>
        where S: DataMut,
    {
        self.ensure_unique();
        let ptr = self.ptr;
        self.dim.stride_offset_checked(&self.strides, &index)
            .map(move |offset| unsafe {
                &mut *ptr.offset(offset)
            })
    }

    /// ***Deprecated: use .get_mut(i)***
    #[cfg_attr(has_deprecated, deprecated(note="use .get_mut() instead"))]
    pub fn at_mut(&mut self, index: D) -> Option<&mut A>
        where S: DataMut,
    {
        self.get_mut(index)
    }

    /// Perform *unchecked* array indexing.
    ///
    /// Return a reference to the element at `index`.
    ///
    /// **Note:** only unchecked for non-debug builds of ndarray.
    #[inline]
    pub unsafe fn uget(&self, index: D) -> &A {
        debug_assert!(self.dim.stride_offset_checked(&self.strides, &index).is_some());
        let off = Dimension::stride_offset(&index, &self.strides);
        &*self.ptr.offset(off)
    }

    /// ***Deprecated: use `.uget()`***
    #[cfg_attr(has_deprecated, deprecated(note="use .uget() instead"))]
    #[inline]
    pub unsafe fn uchk_at(&self, index: D) -> &A {
        self.uget(index)
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
        //debug_assert!(Rc::get_mut(&mut self.data).is_some());
        debug_assert!(self.dim.stride_offset_checked(&self.strides, &index).is_some());
        let off = Dimension::stride_offset(&index, &self.strides);
        &mut *self.ptr.offset(off)
    }

    /// ***Deprecated: use `.uget_mut()`***
    #[cfg_attr(has_deprecated, deprecated(note="use .uget_mut() instead"))]
    #[inline]
    pub unsafe fn uchk_at_mut(&mut self, index: D) -> &mut A
        where S: DataMut
    {
        self.uget_mut(index)
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
    pub fn swap_axes(&mut self, ax: usize, bx: usize)
    {
        self.dim.slice_mut().swap(ax, bx);
        self.strides.slice_mut().swap(ax, bx);
    }

    /// Along `axis`, select the subview `index` and return an
    /// array with that axis removed.
    ///
    /// See [*Subviews*](#subviews) for full documentation.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    ///
    /// ```
    /// use ndarray::{arr1, arr2};
    ///
    /// let a = arr2(&[[1., 2.],    // -- axis 0, row 0
    ///                [3., 4.],    // -- axis 0, row 1
    ///                [5., 6.]]);  // -- axis 0, row 2
    /// //               \   \
    /// //                \   axis 1, column 1
    /// //                 axis 1, column 0
    /// assert!(
    ///     a.subview(0, 1) == arr1(&[3., 4.]) &&
    ///     a.subview(1, 1) == arr1(&[2., 4., 6.])
    /// );
    /// ```
    pub fn subview(&self, axis: usize, index: Ix) -> ArrayBase<S, <D as RemoveAxis>::Smaller>
        where D: RemoveAxis,
              S: DataShared,
    {
        let mut res = self.clone();
        res.isubview(axis, index);
        // don't use reshape -- we always know it will fit the size,
        // and we can use remove_axis on the strides as well
        ArrayBase {
            data: res.data,
            ptr: res.ptr,
            dim: res.dim.remove_axis(axis),
            strides: res.strides.remove_axis(axis),
        }
    }

    /// Collapse dimension `axis` into length one,
    /// and select the subview of `index` along that axis.
    ///
    /// **Panics** if `index` is past the length of the axis.
    pub fn isubview(&mut self, axis: usize, index: Ix)
    {
        dimension::do_sub(&mut self.dim, &mut self.ptr, &self.strides, axis, index)
    }

    /// Along `axis`, select the subview `index` and return a read-write view
    /// with the axis removed.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    ///
    /// ```
    /// use ndarray::{arr2, aview2};
    ///
    /// let mut a = arr2(&[[1., 2.],
    ///                    [3., 4.]]);
    ///
    /// a.subview_mut(1, 1).iadd_scalar(&10.);
    ///
    /// assert!(
    ///     a == aview2(&[[1., 12.],
    ///                   [3., 14.]])
    /// );
    /// ```
    pub fn subview_mut(&mut self, axis: usize, index: Ix)
        -> ArrayViewMut<A, D::Smaller>
        where S: DataMut,
              D: RemoveAxis,
    {
        let mut res = self.view_mut();
        res.isubview(axis, index);
        ArrayBase {
            data: res.data,
            ptr: res.ptr,
            dim: res.dim.remove_axis(axis),
            strides: res.strides.remove_axis(axis),
        }
    }

    /// ***Deprecated: use `.subview_mut()`***
    #[cfg_attr(has_deprecated, deprecated(note="use .subview_mut() instead"))]
    pub fn sub_iter_mut(&mut self, axis: usize, index: Ix)
        -> ElementsMut<A, D>
        where S: DataMut,
    {
        let mut it = self.view_mut();
        dimension::do_sub(&mut it.dim, &mut it.ptr, &it.strides, axis, index);
        it.into_iter_()
    }

    /// Return an iterator that traverses over all dimensions but the innermost,
    /// and yields each inner row.
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
        iterators::new_outer(self.view())
    }

    /// Return an iterator that traverses over all dimensions but the innermost,
    /// and yields each inner row.
    ///
    /// Iterator element is `ArrayViewView<A, Ix>` (1D read-write array view).
    pub fn inner_iter_mut(&mut self) -> InnerIterMut<A, D>
        where S: DataMut
    {
        iterators::new_outer_mut(self.view_mut())
    }

    // Return (length, stride) for diagonal
    fn diag_params(&self) -> (Ix, Ixs)
    {
        /* empty shape has len 1 */
        let len = self.dim.slice().iter().map(|x| *x).min().unwrap_or(1);
        let stride = self.strides.slice().iter()
                        .map(|x| *x as Ixs)
                        .fold(0, |sum, s| sum + s);
        return (len, stride)
    }

    /// Return an iterator over the diagonal elements of the array.
    ///
    /// The diagonal is simply the sequence indexed by *(0, 0, .., 0)*,
    /// *(1, 1, ..., 1)* etc as long as all axes have elements.
    pub fn diag_iter(&self) -> Elements<A, Ix>
    {
        let (len, stride) = self.diag_params();
        let view = ArrayBase {
            data: self.raw_data(),
            ptr: self.ptr,
            dim: len,
            strides: stride as Ix,
        };
        view.into_iter_()
    }

    /// Return the diagonal as a one-dimensional array.
    pub fn diag(&self) -> ArrayBase<S, Ix>
        where S: DataShared,
    {
        let (len, stride) = self.diag_params();
        ArrayBase {
            data: self.data.clone(),
            ptr: self.ptr,
            dim: len,
            strides: stride as Ix,
        }
    }

    /// Return a read-write view over the diagonal elements of the array.
    pub fn diag_mut(&mut self) -> ArrayViewMut<A, Ix>
        where S: DataMut,
    {
        self.ensure_unique();
        let (len, stride) = self.diag_params();
        ArrayViewMut {
            ptr: self.ptr,
            data: self.raw_data_mut(),
            dim: len,
            strides: stride as Ix,
        }
    }

    /// ***Deprecated: use `.diag_mut()`***
    #[cfg_attr(has_deprecated, deprecated(note="use .diag_mut() instead"))]
    pub fn diag_iter_mut(&mut self) -> ElementsMut<A, Ix>
        where S: DataMut,
    {
        self.diag_mut().into_iter_()
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

    /*
    /// Set the array to the standard layout, without adjusting elements.
    /// Useful for overwriting.
    fn force_standard_layout(&mut self) {
        self.strides = self.dim.default_strides();
    }
    */
    /// Return `true` if the array data is laid out in contiguous “C order” in
    /// memory (where the last index is the most rapidly varying).
    ///
    /// Return `false` otherwise, i.e the array is possibly not
    /// contiguous in memory, it has custom strides, etc.
    pub fn is_standard_layout(&self) -> bool
    {
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
    /// use ndarray::{arr1, arr2};
    ///
    /// assert!(
    ///     arr1(&[1., 2., 3., 4.]).reshape((2, 2))
    ///     == arr2(&[[1., 2.],
    ///               [3., 4.]])
    /// );
    /// ```
    pub fn reshape<E: Dimension>(&self, shape: E) -> ArrayBase<S, E>
        where S: DataShared + DataOwned, A: Clone,
    {
        if shape.size() != self.dim.size() {
            panic!("Incompatible shapes in reshape, attempted from: {:?}, to: {:?}",
                   self.dim.slice(), shape.slice())
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
                ArrayBase::from_vec_dim(shape, v)
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
        if shape.size() != self.dim.size() {
            return Err(ShapeError::IncompatibleShapes(
                    self.dim.slice().to_vec().into_boxed_slice(),
                    shape.slice().to_vec().into_boxed_slice()));
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
    /// use ndarray::arr1;
    ///
    /// assert!(
    ///     arr1(&[1., 0.]).broadcast((10, 2)).unwrap().dim()
    ///     == (10, 2)
    /// );
    /// ```
    pub fn broadcast<E>(&self, dim: E)
        -> Option<ArrayView<A, E>>
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
                return None
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
        let broadcast_strides =
            match upcast(&dim, &self.dim, &self.strides) {
                Some(st) => st,
                None => return None,
            };
        Some(ArrayView {
            data: self.raw_data(),
            ptr: self.ptr,
            dim: dim,
            strides: broadcast_strides,
        })
    }

    #[cfg_attr(has_deprecated, deprecated(note="use .broadcast() instead"))]
    /// ***Deprecated: Use `.broadcast()` instead.***
    pub fn broadcast_iter<E>(&self, dim: E) -> Option<Elements<A, E>>
        where E: Dimension,
    {
        self.broadcast(dim).map(|v| v.into_iter_())
    }

    #[inline]
    fn broadcast_unwrap<E>(&self, dim: E) -> ArrayView<A, E>
        where E: Dimension,
    {
        match self.broadcast(dim.clone()) {
            Some(it) => it,
            None => Self::broadcast_panic(&self.dim, &dim),
        }
    }

    #[inline(never)]
    fn broadcast_panic<E: Dimension>(from: &D, to: &E) -> ! {
        panic!("Could not broadcast array from shape: {:?} to: {:?}",
               from.slice(), to.slice())
    }

    /// Return a slice of the array’s backing data in memory order.
    ///
    /// **Note:** Data memory order may not correspond to the index order
    /// of the array. Neither is the raw data slice is restricted to just the
    /// Array’s view.<br>
    /// **Note:** the slice may be empty.
    pub fn raw_data(&self) -> &[A] {
        self.data.slice()
    }

    /// Return a mutable slice of the array’s backing data in memory order.
    ///
    /// **Note:** Data memory order may not correspond to the index order
    /// of the array. Neither is the raw data slice is restricted to just the
    /// Array’s view.<br>
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
            // FIXME: Regular .zip() is slow
            for (y, x) in s_row.iter_mut().zip(r_row) {
                f(y, x);
            }
        }
    }

    // FIXME: Guarantee the order here or not?
    /// Traverse two arrays in unspecified order, in lock step,
    /// calling the closure `f` on each element pair.
    ///
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting isn’t possible.
    #[inline]
    pub fn zip_mut_with<B, S2, E, F>(&mut self, rhs: &ArrayBase<S2, E>, mut f: F)
        where S: DataMut,
              S2: Data<Elem=B>,
              E: Dimension,
              F: FnMut(&mut A, &B)
    {
        if self.dim.ndim() == rhs.dim.ndim() && self.shape() == rhs.shape() {
            self.zip_with_mut_same_shape(rhs, f);
        } else if rhs.dim.ndim() == 0 {
            // Skip broadcast from 0-dim array
            // FIXME: Order
            unsafe {
                let rhs_elem = &*rhs.ptr;
                let f_ = &mut f;
                self.unordered_foreach_mut(move |elt| f_(elt, rhs_elem));
            }
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
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert!(
    ///     a.map(|&x| (x / 2.) as i32)
    ///     == arr2(&[[0, 1], [1, 2]])
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
            ArrayBase::from_vec_dim(self.dim.clone(), res)
        }
    }
}

/// Return an array filled with zeros
pub fn zeros<A, D>(dim: D) -> OwnedArray<A, D>
    where A: Clone + libnum::Zero, D: Dimension,
{
    ArrayBase::zeros(dim)
}

/// Return a zero-dimensional array with the element `x`.
pub fn arr0<A>(x: A) -> Array<A, ()>
{
    unsafe { Array::from_vec_dim((), vec![x]) }
}

/// Return a one-dimensional array with elements from `xs`.
pub fn arr1<A: Clone>(xs: &[A]) -> Array<A, Ix>
{
    Array::from_vec(xs.to_vec())
}

/// Return a zero-dimensional array view borrowing `x`.
pub fn aview0<A>(x: &A) -> ArrayView<A, ()> {
    let data = unsafe {
        std::slice::from_raw_parts(x, 1)
    };
    ArrayView {
        data: data,
        ptr: data.as_ptr() as *mut _,
        dim: (),
        strides: (),
    }
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
    ArrayView {
        data: xs,
        ptr: xs.as_ptr() as *mut _,
        dim: xs.len() as Ix,
        strides: 1,
    }
}

/// Return a two-dimensional array view with elements borrowing `xs`.
pub fn aview2<A, V: FixedInitializer<Elem=A>>(xs: &[V]) -> ArrayView<A, (Ix, Ix)> {
    let cols = V::len();
    let rows = xs.len();
    let data = unsafe {
        std::slice::from_raw_parts(xs.as_ptr() as *const A, cols * rows)
    };
    let dim = (rows as Ix, cols as Ix);
    ArrayView {
        data: data,
        ptr: data.as_ptr() as *mut _,
        strides: dim.default_strides(),
        dim: dim,
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
    ArrayViewMut {
        ptr: xs.as_mut_ptr(),
        dim: xs.len() as Ix,
        strides: 1,
        data: xs,
    }
}

/// Slice or fixed-size array used for array initialization
pub unsafe trait Initializer {
    type Elem;
    fn as_init_slice(&self) -> &[Self::Elem];
    fn is_fixed_size() -> bool { false }
}

/// Fixed-size array used for array initialization
pub unsafe trait FixedInitializer: Initializer {
    fn len() -> usize;
}

unsafe impl<T> Initializer for [T] {
    type Elem = T;
    fn as_init_slice(&self) -> &[T] {
        self
    }
}

macro_rules! impl_arr_init {
    (__impl $n: expr) => (
        unsafe impl<T> Initializer for [T;  $n] {
            type Elem = T;
            fn as_init_slice(&self) -> &[T] { self }
            fn is_fixed_size() -> bool { true }
        }

        unsafe impl<T> FixedInitializer for [T;  $n] {
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
/// **Panics** if the slices are not all of the same length.
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
pub fn arr2<A: Clone, V: Initializer<Elem=A>>(xs: &[V]) -> Array<A, (Ix, Ix)>
{
    // FIXME: Simplify this when V is fix size array
    let (m, n) = (xs.len() as Ix,
                  xs.get(0).map_or(0, |snd| snd.as_init_slice().len() as Ix));
    let dim = (m, n);
    let mut result = Vec::<A>::with_capacity(dim.size());
    for snd in xs.iter() {
        let snd = snd.as_init_slice();
        assert!(<V as Initializer>::is_fixed_size() || snd.len() as Ix == n);
        result.extend(snd.iter().map(|x| x.clone()))
    }
    unsafe {
        Array::from_vec_dim(dim, result)
    }
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
pub fn arr3<A: Clone, V: Initializer<Elem=U>, U: Initializer<Elem=A>>(xs: &[V])
    -> Array<A, (Ix, Ix, Ix)>
{
    // FIXME: Simplify this when U/V are fix size arrays
    let m = xs.len() as Ix;
    let fst = xs.get(0).map(|snd| snd.as_init_slice());
    let thr = fst.and_then(|elt| elt.get(0).map(|elt2| elt2.as_init_slice()));
    let n = fst.map_or(0, |v| v.len() as Ix);
    let o = thr.map_or(0, |v| v.len() as Ix);
    let dim = (m, n, o);
    let mut result = Vec::<A>::with_capacity(dim.size());
    for snd in xs.iter() {
        let snd = snd.as_init_slice();
        assert!(<V as Initializer>::is_fixed_size() || snd.len() as Ix == n);
        for thr in snd.iter() {
            let thr = thr.as_init_slice();
            assert!(<U as Initializer>::is_fixed_size() || thr.len() as Ix == o);
            result.extend(thr.iter().map(|x| x.clone()))
        }
    }
    unsafe {
        Array::from_vec_dim(dim, result)
    }
}


impl<A, S, D> ArrayBase<S, D>
    where S: Data<Elem=A>,
          D: Dimension,
{
    /// Return sum along `axis`.
    ///
    /// ```
    /// use ndarray::{aview0, aview1, arr2};
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert!(
    ///     a.sum(0) == aview1(&[4., 6.]) &&
    ///     a.sum(1) == aview1(&[3., 7.]) &&
    ///
    ///     a.sum(0).sum(0) == aview0(&10.)
    /// );
    /// ```
    ///
    /// **Panics** if `axis` is out of bounds.
    pub fn sum(&self, axis: usize) -> OwnedArray<A, <D as RemoveAxis>::Smaller>
        where A: Clone + Add<Output=A>,
              D: RemoveAxis,
    {
        let n = self.shape()[axis];
        let mut res = self.view().subview(axis, 0).to_owned();
        for i in 1..n {
            let view = self.view().subview(axis, i);
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
            return Self::unrolled_sum(slc);
        }
        let mut sum = A::zero();
        for row in self.inner_iter() {
            if let Some(slc) = row.as_slice() {
                sum = sum + Self::unrolled_sum(slc);
            } else {
                sum = sum + row.fold(A::zero(), |acc, elt| acc + elt.clone());
            }
        }
        sum
    }

    fn unrolled_sum(mut xs: &[A]) -> A
        where A: Clone + Add<Output=A> + libnum::Zero,
    {
        // eightfold unrolled so that floating point can be vectorized
        // (even with strict floating point accuracy semantics)
        let mut sum = A::zero();
        let (mut p0, mut p1, mut p2, mut p3,
             mut p4, mut p5, mut p6, mut p7) =
            (A::zero(), A::zero(), A::zero(), A::zero(),
             A::zero(), A::zero(), A::zero(), A::zero());
        while xs.len() >= 8 {
            p0 = p0 + xs[0].clone();
            p1 = p1 + xs[1].clone();
            p2 = p2 + xs[2].clone();
            p3 = p3 + xs[3].clone();
            p4 = p4 + xs[4].clone();
            p5 = p5 + xs[5].clone();
            p6 = p6 + xs[6].clone();
            p7 = p7 + xs[7].clone();

            xs = &xs[8..];
        }
        sum = sum.clone() + (p0 + p4);
        sum = sum.clone() + (p1 + p5);
        sum = sum.clone() + (p2 + p6);
        sum = sum.clone() + (p3 + p7);
        for elt in xs {
            sum = sum.clone() + elt.clone();
        }
        sum
    }

    /// Return mean along `axis`.
    ///
    /// ```
    /// use ndarray::{aview1, arr2};
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert!(
    ///     a.mean(0) == aview1(&[2.0, 3.0]) &&
    ///     a.mean(1) == aview1(&[1.5, 3.5])
    /// );
    /// ```
    ///
    ///
    /// **Panics** if `axis` is out of bounds.
    #[allow(deprecated)]
    pub fn mean(&self, axis: usize) -> OwnedArray<A, <D as RemoveAxis>::Smaller>
        where A: Copy + Field,
              D: RemoveAxis,
    {
        let n = self.shape()[axis];
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
        where A: Float + PartialOrd,
              S2: Data<Elem=A>,
    {
        self.shape() == rhs.shape() &&
        self.iter().zip(rhs.iter()).all(|(x, y)| (*x - *y).abs() <= tol)
    }
}

impl<A, S> ArrayBase<S, (Ix, Ix)>
    where S: Data<Elem=A>,
{
    unsafe fn one_dimensional_iter<'a>(ptr: *mut A, len: Ix, stride: Ix)
        -> Elements<'a, A, Ix>
    {
        // NOTE: `data` field is unused by into_iter
        let view = ArrayView {
            data: &[],
            ptr: ptr,
            dim: len,
            strides: stride,
        };
        view.into_iter_()
    }

    /// Return an iterator over the elements of row `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn row_iter(&self, index: Ix) -> Elements<A, Ix>
    {
        let (m, n) = self.dim;
        let (sr, sc) = self.strides;
        assert!(index < m);
        unsafe {
            Self::one_dimensional_iter(self.ptr.offset(stride_offset(index, sr)), n, sc)
        }
    }

    /// Return an iterator over the elements of column `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn col_iter(&self, index: Ix) -> Elements<A, Ix>
    {
        let (m, n) = self.dim;
        let (sr, sc) = self.strides;
        assert!(index < n);
        unsafe {
            Self::one_dimensional_iter(self.ptr.offset(stride_offset(index, sc)), m, sr)
        }
    }

    /// Perform matrix multiplication of rectangular arrays `self` and `rhs`.
    ///
    /// The array sizes must agree in the way that
    /// if `self` is *M* × *N*, then `rhs` is *N* × *K*.
    ///
    /// Return a result array with shape *M* × *K*.
    ///
    /// **Panics** if sizes are incompatible.
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
    #[allow(deprecated)]
    pub fn mat_mul(&self, rhs: &ArrayBase<S, (Ix, Ix)>) -> Array<A, (Ix, Ix)>
        where A: Copy + Ring
    {
        // NOTE: Matrix multiplication only defined for simple types to
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
        for rr in res_elems.iter_mut() {
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
            ArrayBase::from_vec_dim((m, n), res_elems)
        }
    }

    /// Perform the matrix multiplication of the rectangular array `self` and
    /// column vector `rhs`.
    ///
    /// The array sizes must agree in the way that
    /// if `self` is *M* × *N*, then `rhs` is *N*.
    ///
    /// Return a result array with shape *M*.
    ///
    /// **Panics** if sizes are incompatible.
    #[allow(deprecated)]
    pub fn mat_mul_col(&self, rhs: &ArrayBase<S, Ix>) -> Array<A, Ix>
        where A: Copy + Ring
    {
        let ((m, a), n) = (self.dim, rhs.dim);
        let (self_columns, other_rows) = (a, n);
        assert!(self_columns == other_rows);

        // Avoid initializing the memory in vec -- set it during iteration
        let mut res_elems = Vec::<A>::with_capacity(m as usize);
        unsafe {
            res_elems.set_len(m as usize);
        }
        let mut i = 0;
        for rr in res_elems.iter_mut() {
            unsafe {
                *rr = (0..a).fold(libnum::zero::<A>(),
                    move |s, k| s + *self.uget((i, k)) * *rhs.uget(k)
                );
            }
            i += 1;
        }
        unsafe {
            ArrayBase::from_vec_dim(m, res_elems)
        }
    }
}



// Array OPERATORS

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

macro_rules! impl_binary_op(
    ($trt:ident, $mth:ident, $doc:expr) => (
/// Perform elementwise
#[doc=$doc]
/// between `self` and `rhs`,
/// and return the result (based on `self`).
///
/// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
///
/// **Panics** if broadcasting isn’t possible.
impl<A, S, S2, D, E> $trt<ArrayBase<S2, E>> for ArrayBase<S, D>
    where A: Clone + $trt<A, Output=A>,
          S: DataMut<Elem=A>,
          S2: Data<Elem=A>,
          D: Dimension,
          E: Dimension,
{
    type Output = ArrayBase<S, D>;
    fn $mth (mut self, rhs: ArrayBase<S2, E>) -> ArrayBase<S, D>
    {
        // FIXME: Can we co-broadcast arrays here? And how?
        self.zip_mut_with(&rhs, |x, y| {
            *x = x.clone(). $mth (y.clone());
        });
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
        self.to_owned().$mth(rhs.view())
    }
}
    );
);

mod arithmetic_ops {
    use super::*;
    use std::ops::*;

    impl_binary_op!(Add, add, "addition");
    impl_binary_op!(Sub, sub, "subtraction");
    impl_binary_op!(Mul, mul, "multiplication");
    impl_binary_op!(Div, div, "division");
    impl_binary_op!(Rem, rem, "remainder");
    impl_binary_op!(BitAnd, bitand, "bit and");
    impl_binary_op!(BitOr, bitor, "bit or");
    impl_binary_op!(BitXor, bitxor, "bit xor");
    impl_binary_op!(Shl, shl, "left shift");
    impl_binary_op!(Shr, shr, "right shift");

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

    use std::ops::{
        AddAssign,
        SubAssign,
        MulAssign,
        DivAssign,
        RemAssign,
        BitAndAssign,
        BitOrAssign,
        BitXorAssign,
    };


    macro_rules! impl_assign_op {
        ($trt:ident, $method:ident, $doc:expr) => {

    #[doc=$doc]
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting isn’t possible.
    ///
    /// **Requires `feature = "assign_ops"`**
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
#[derive(Clone)]
pub struct Indexed<'a, A: 'a, D>(ElementsBase<'a, A, D>);
/// An iterator over the indexes and elements of an array (mutable).
pub struct IndexedMut<'a, A: 'a, D>(ElementsBaseMut<'a, A, D>);

fn zipsl<T, U>(t: T, u: U) -> ZipSlices<T, U>
    where T: it::misc::Slice, U: it::misc::Slice
{
    ZipSlices::from_slices(t, u)
}

enum ElementsRepr<S, C> {
    Slice(S),
    Counted(C),
}
