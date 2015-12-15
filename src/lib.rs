#![crate_name="ndarray"]
#![crate_type="dylib"]

//! The `ndarray` crate provides an n-dimensional container similar to numpy's ndarray.
//!
//! - [`ArrayBase`](struct.ArrayBase.html)<br>
//!   The n-dimensional array type itself, parameterized by data storage.
//! - `Array`<br>
//!   Array where the data is reference counted and copy on write, it
//!   can act as both an owner as the data as well as a lightweight view.
//! - `OwnedArray`<br>
//!   Array where the data is owned uniquely.
//! - `ArrayView`<br>
//!   A lightweight array view.
//! - `ArrayViewMut`<br>
//!   A lightweight read-write array view.
//!
//! ## Crate Summary and Status
//!
//! - Implements the numpy striding scheme for n-dimensional arrays
//! - `Array` is clone on write, so it can be both a view or an owner of the
//!   data.
//! - Striding and broadcasting is fully implemented
//! - Focus is on being a generic n-dimensional container
//! - Due to iterators, arithmetic operations, matrix multiplication etc
//!   are not very well optimized, this is not a serious crate for numerics
//!   or linear algebra. `Array` is a good container.
//! - There is no integration with linear algebra packages (at least not yet).
//!
//! ## Crate feature flags
//!
//! - `assign_ops`
//!   - Optional, requires nightly
//!   - Enables the compound assignment operators
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

use std::mem;
use std::rc::Rc;
use libnum::Float;
use std::ops::{Add, Sub, Mul, Div, Rem, Neg, Not, Shr, Shl,
    BitAnd,
    BitOr,
    BitXor,
};

pub use dimension::{Dimension, RemoveAxis};
pub use si::{Si, S, SliceRange};
use dimension::stride_offset;

pub use indexes::Indexes;

use iterators::Baseiter;


pub mod linalg;
mod arraytraits;
#[cfg(feature = "serde")]
mod arrayserialize;
mod arrayformat;
mod dimension;
mod indexes;
mod iterators;
mod si;
//mod macros;

// NOTE: In theory, the whole library should compile
// and pass tests even if you change Ix and Ixs.
/// Array index type
pub type Ix = u32;
/// Array index type (signed)
pub type Ixs = i32;

/// The `Array` type is an *N-dimensional array*.
///
/// A reference counted array with copy-on-write mutability.
///
/// The array can be a container of numerical use, supporting
/// all mathematical operators by applying them elementwise -- but it can
/// store any kind of value. It cannot grow or shrink, but can be sliced into
/// views of parts of its data.
///
/// The array is both a view and a shared owner of its data. Some methods,
/// for example [*slice()*](#method.slice), merely change the view of the data,
/// while methods like [*iadd()*](#method.iadd) allow mutating the element
/// values.
///
/// Calling a method for mutating elements, for example 
/// [*get_mut()*](#method.get_mut), [*iadd()*](#method.iadd) or
/// [*iter_mut()*](#method.iter_mut) will break sharing and require a clone of
/// the data (if it is not uniquely held).
///
/// ## Method Conventions
///
/// Methods mutating the view or array elements in place use an *i* prefix,
/// for example *slice* vs. *islice* and *add* vs *iadd*.
///
/// ## Indexing
///
/// Arrays use `u32` for indexing, represented by the types `Ix` and `Ixs` 
/// (signed).
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

/// Array's inner representation.
pub unsafe trait Data {
    type Elem;
    fn slice(&self) -> &[Self::Elem];
}

/// Array's writable inner representation.
pub unsafe trait DataMut : Data {
    fn slice_mut(&mut self) -> &mut [Self::Elem];
    fn ensure_unique<D>(&mut ArrayBase<Self, D>)
        where Self: Sized, D: Dimension
    {
    }
}

/// Clone an Array's storage.
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
pub trait DataShared : Clone + DataClone { }

impl<A> DataShared for Rc<Vec<A>> { }
impl<'a, A> DataShared for &'a [A] { }

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

impl<S> ArrayBase<S, Ix>
    where S: DataOwned,
{
    /// Create a one-dimensional array from a vector (no allocation needed).
    pub fn from_vec(v: Vec<S::Elem>) -> Self {
        unsafe {
            Self::from_vec_dim(v.len() as Ix, v)
        }
    }

    /// Create a one-dimensional array from an iterable.
    pub fn from_iter<I: IntoIterator<Item=S::Elem>>(iterable: I) -> ArrayBase<S, Ix> {
        Self::from_vec(iterable.into_iter().collect())
    }
}

impl Array<f32, Ix>
{
    /// Create a one-dimensional Array from interval `[begin, end)`
    pub fn range(begin: f32, end: f32) -> Array<f32, Ix>
    {
        let n = (end - begin) as usize;
        let span = if n > 0 { (n - 1) as f32 } else { 0. };
        Array::from_iter(it::linspace(begin,
                                      begin + span,
                                      n))
    }
}

impl<S, D> ArrayBase<S, D>
    where S: DataOwned,
          D: Dimension,
{
    /// Create an array from a vector (with no allocation needed).
    ///
    /// Unsafe because dimension is unchecked, and must be correct.
    pub unsafe fn from_vec_dim(dim: D, mut v: Vec<S::Elem>) -> ArrayBase<S, D>
    {
        debug_assert!(dim.size() == v.len());
        ArrayBase {
            ptr: v.as_mut_ptr(),
            data: DataOwned::new(v),
            strides: dim.default_strides(),
            dim: dim
        }
    }

    /// Construct an Array with zeros.
    pub fn zeros(dim: D) -> ArrayBase<S, D> where S::Elem: Clone + libnum::Zero
    {
        Self::from_elem(dim, libnum::zero())
    }

    /// Construct an Array with copies of `elem`.
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
    pub fn from_elem(dim: D, elem: S::Elem) -> ArrayBase<S, D> where S::Elem: Clone
    {
        let v = std::iter::repeat(elem).take(dim.size()).collect();
        unsafe {
            Self::from_vec_dim(dim, v)
        }
    }

    /// Construct an Array with default values, dimension `dim`.
    pub fn default(dim: D) -> ArrayBase<S, D>
        where S::Elem: Default
    {
        let v = (0..dim.size()).map(|_| <S::Elem>::default()).collect();
        unsafe {
            Self::from_vec_dim(dim, v)
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

    fn into_iter_(self) -> Elements<'a, A, D> {
        Elements { inner: self.into_base_iter() }
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

    fn into_iter_(self) -> ElementsMut<'a, A, D> {
        ElementsMut { inner: self.into_base_iter() }
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

    /// Return `true` if the array data is laid out in
    /// contiguous “C order” where the last index is the most rapidly
    /// varying.
    ///
    /// Return `false` otherwise, i.e the array is possibly not
    /// contiguous in memory, it has custom strides, etc.
    pub fn is_standard_layout(&self) -> bool
    {
        self.strides == self.dim.default_strides()
    }

    /// Return a read-only view of the array
    pub fn view(&self) -> ArrayView<A, D> {
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

    /// Return an uniquely owned copy of the array or view
    pub fn to_owned(&self) -> OwnedArray<A, D>
        where A: Clone
    {
        // FIXME: Use standard layout / more efficient copy?
        let data = self.iter().cloned().collect();
        unsafe {
            ArrayBase::from_vec_dim(self.dim.clone(), data)
        }
    }

    /// Return a shared ownership (copy on write) array.
    pub fn to_shared(&self) -> Array<A, D>
        where A: Clone
    {
        // FIXME: Avoid copying if it's already an Array.
        // FIXME: Use standard layout / more efficient copy?
        let data = self.iter().cloned().collect();
        unsafe {
            ArrayBase::from_vec_dim(self.dim.clone(), data)
        }
    }

    /// Return a shared ownership (copy on write) array.
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

    /// Return a slice of the array's backing data in memory order.
    ///
    /// **Note:** Data memory order may not correspond to the index order
    /// of the array. Neither is the raw data slice is restricted to just the
    /// Array's view.
    pub fn raw_data(&self) -> &[A]
    {
        self.data.slice()
    }

    /// Return a sliced array.
    ///
    /// **Panics** if `indexes` does not match the number of array axes.
    pub fn slice(&self, indexes: &[Si]) -> Self
        where S: DataShared
    {
        let mut arr = self.clone();
        arr.islice(indexes);
        arr
    }

    /// Slice the array's view in place.
    ///
    /// **Panics** if `indexes` does not match the number of array axes.
    pub fn islice(&mut self, indexes: &[Si])
    {
        let offset = Dimension::do_slices(&mut self.dim, &mut self.strides, indexes);
        unsafe {
            self.ptr = self.ptr.offset(offset);
        }
    }

    /// Return an iterator over a sliced view.
    ///
    /// **Panics** if `indexes` does not match the number of array axes.
    pub fn slice_iter(&self, indexes: &[Si]) -> Elements<A, D>
    {
        let mut it = self.iter();
        let offset = Dimension::do_slices(&mut it.inner.dim, &mut it.inner.strides, indexes);
        unsafe {
            it.inner.ptr = it.inner.ptr.offset(offset);
        }
        it
    }

    /// Return a reference to the element at `index`, or return `None` 
    /// if the index is out of bounds.
    pub fn get(&self, index: D) -> Option<&A> {
        self.dim.stride_offset_checked(&self.strides, &index)
            .map(|offset| unsafe {
                &*self.ptr.offset(offset)
            })
    }

    /// ***Deprecated: use .get(i)***
    pub fn at(&self, index: D) -> Option<&A> {
        self.get(index)
    }

    /// Return a mutable reference to the element at `index`, or return `None`
    /// if the index is out of bounds.
    pub fn get_mut(&mut self, index: D) -> Option<&mut A>
        where S: DataMut,
    {
        self.ensure_unique();
        self.dim.stride_offset_checked(&self.strides, &index)
            .map(|offset| unsafe {
                &mut *self.ptr.offset(offset)
            })
    }

    /// ***Deprecated: use .get_mut(i)***
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
    pub unsafe fn uchk_at(&self, index: D) -> &A {
        debug_assert!(self.dim.stride_offset_checked(&self.strides, &index).is_some());
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
    pub unsafe fn uchk_at_mut(&mut self, index: D) -> &mut A
        where S: DataMut
    {
        //debug_assert!(Rc::get_mut(&mut self.data).is_some());
        debug_assert!(self.dim.stride_offset_checked(&self.strides, &index).is_some());
        let off = Dimension::stride_offset(&index, &self.strides);
        &mut *self.ptr.offset(off)
    }

    /// Return a protoiterator
    #[inline]
    fn base_iter<'a>(&'a self) -> Baseiter<'a, A, D>
    {
        unsafe {
            Baseiter::new(self.ptr, self.dim.clone(), self.strides.clone())
        }
    }

    /// Return an iterator of references to the elements of the array.
    ///
    /// Iterator element type is `&A`.
    pub fn iter(&self) -> Elements<A, D>
    {
        Elements { inner: self.base_iter() }
    }

    /// Return an iterator of references to the elements of the array.
    ///
    /// Iterator element type is `(D, &A)`.
    pub fn indexed_iter(&self) -> Indexed<Elements<A, D>>
    {
        self.iter().indexed()
    }

    /// Collapse dimension `axis` into length one,
    /// and select the subview of `index` along that axis.
    ///
    /// **Panics** if `index` is past the length of the axis.
    pub fn isubview(&mut self, axis: usize, index: Ix)
    {
        dimension::do_sub(&mut self.dim, &mut self.ptr, &self.strides, axis, index)
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
    pub fn broadcast<E: Dimension>(&self, dim: E)
        -> Option<ArrayView<A, E>>
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

    /// Act like a larger size and/or shape array by *broadcasting*
    /// into a larger shape, if possible.
    ///
    /// Return `None` if shapes can not be broadcast together.
    pub fn broadcast_iter<E: Dimension>(&self, dim: E)
        -> Option<Elements<A, E>>
    {
        self.broadcast(dim).map(|v| v.into_iter_())
    }

    #[inline(never)]
    fn broadcast_iter_unwrap<E: Dimension>(&self, dim: E)
        -> Elements<A, E>
    {
        match self.broadcast_iter(dim.clone()) {
            Some(it) => it,
            None => panic!("Could not broadcast array from shape {:?} into: {:?}",
                           self.shape(), dim.slice())
        }
    }

    /// Swap axes `ax` and `bx`.
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
        unsafe {
            Elements { inner:
                Baseiter::new(self.ptr, len, stride as Ix)
            }
        }
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

    /// Select the subview `index` along `axis` and return an
    /// array with that axis removed.
    ///
    /// **Panics** if `index` is past the length of the axis.
    ///
    /// ```
    /// use ndarray::{arr1, arr2};
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    ///
    /// assert!(
    ///     a.subview(0, 0) == arr1(&[1., 2.]) &&
    ///     a.subview(1, 1) == arr1(&[2., 4.])
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

    /// Make the array unshared.
    ///
    /// This method is mostly only useful with unsafe code.
    fn ensure_unique(&mut self)
        where S: DataMut
    {
        S::ensure_unique(self);
    }

    /// Return an iterator of mutable references to the elements of the array.
    ///
    /// Iterator element type is `&mut A`.
    pub fn iter_mut(&mut self) -> ElementsMut<A, D>
        where S: DataMut,
    {
        self.ensure_unique();
        ElementsMut { inner: self.base_iter() }
    }

    /// Return an iterator of indexes and mutable references to the elements of the array.
    ///
    /// Iterator element type is `(D, &mut A)`.
    pub fn indexed_iter_mut(&mut self) -> Indexed<ElementsMut<A, D>>
        where S: DataMut,
    {
        self.iter_mut().indexed()
    }

    /// Return a sliced read-write view of the array.
    ///
    /// **Panics** if `indexes` does not match the number of array axes.
    pub fn slice_mut(&mut self, indexes: &[Si]) -> ArrayViewMut<A, D>
        where S: DataMut
    {
        let mut arr = self.view_mut();
        arr.islice(indexes);
        arr
    }

    /// Return an iterator of mutable references into the sliced view
    /// of the array.
    ///
    /// Iterator element type is `&mut A`.
    ///
    /// **Panics** if `indexes` does not match the number of array axes.
    pub fn slice_iter_mut(&mut self, indexes: &[Si]) -> ElementsMut<A, D>
        where S: DataMut,
    {
        let mut it = self.iter_mut();
        let offset = Dimension::do_slices(&mut it.inner.dim, &mut it.inner.strides, indexes);
        unsafe {
            it.inner.ptr = it.inner.ptr.offset(offset);
        }
        it
    }

    /// Select the subview `index` along `axis` and return a read-write view.
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

    /// Select the subview `index` along `axis` and return an iterator
    /// of the subview.
    ///
    /// Iterator element type is `&mut A`.
    ///
    /// **Panics** if `axis` or `index` is out of bounds.
    pub fn sub_iter_mut(&mut self, axis: usize, index: Ix)
        -> ElementsMut<A, D>
        where S: DataMut,
    {
        let mut it = self.iter_mut();
        dimension::do_sub(&mut it.inner.dim, &mut it.inner.ptr, &it.inner.strides, axis, index);
        it
    }

    /// Return an iterator over the diagonal elements of the array.
    pub fn diag_iter_mut(&mut self) -> ElementsMut<A, Ix>
        where S: DataMut,
    {
        self.ensure_unique();
        let (len, stride) = self.diag_params();
        unsafe {
            ElementsMut { inner:
                Baseiter::new(self.ptr, len, stride as Ix),
            }
        }
    }

    /// Return a mutable slice of the array's backing data in memory order.
    ///
    /// **Note:** Data memory order may not correspond to the index order
    /// of the array. Neither is the raw data slice is restricted to just the
    /// array's view.
    ///
    /// **Note:** The data is uniquely held and nonaliased
    /// while it is mutably borrowed.
    pub fn raw_data_mut(&mut self) -> &mut [A]
        where S: DataMut,
    {
        self.ensure_unique();
        self.data.slice_mut()
    }


    /// Transform the array into `shape`; any other shape
    /// with the same number of elements is accepted.
    ///
    /// **Panics** if sizes are incompatible.
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
            panic!("Incompatible sizes in reshape, attempted from: {:?}, to: {:?}",
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

    /// Perform an elementwise assigment to `self` from `rhs`.
    ///
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting isn't possible.
    pub fn assign<E: Dimension, S2>(&mut self, rhs: &ArrayBase<S2, E>)
        where S: DataMut,
              A: Clone,
              S2: Data<Elem=A>,
    {
        if self.shape() == rhs.shape() {
            for (x, y) in self.iter_mut().zip(rhs.iter()) {
                *x = y.clone();
            }
        } else {
            let other_iter = rhs.broadcast_iter_unwrap(self.dim());
            for (x, y) in self.iter_mut().zip(other_iter) {
                *x = y.clone();
            }
        }
    }

    /// Perform an elementwise assigment to `self` from scalar `x`.
    pub fn assign_scalar(&mut self, x: &A)
        where S: DataMut, A: Clone,
    {
        for elt in self.iter_mut() {
            *elt = x.clone();
        }
    }
}

/// Return a zero-dimensional array with the element `x`.
pub fn arr0<A>(x: A) -> Array<A, ()>
{
    let mut v = Vec::with_capacity(1);
    v.push(x);
    unsafe { Array::from_vec_dim((), v) }
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
    where A: Clone + Add<Output=A>,
          S: Data<Elem=A>,
          D: RemoveAxis,
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
    {
        let n = self.shape()[axis];
        let mut res = self.view().subview(axis, 0).to_owned();
        for i in 1..n {
            let view = self.view().subview(axis, i);
            res.iadd(&view);
        }
        res
    }
}

impl<A, S, D> ArrayBase<S, D>
    where A: Copy + linalg::Field,
          S: Data<Elem=A>,
          D: RemoveAxis,
{
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
    pub fn mean(&self, axis: usize) -> OwnedArray<A, <D as RemoveAxis>::Smaller>
    {
        let n = self.shape()[axis];
        let mut sum = self.sum(axis);
        let one = libnum::one::<A>();
        let mut cnt = one;
        for _ in 1..n {
            cnt = cnt + one;
        }
        for elt in sum.iter_mut() {
            *elt = *elt / cnt;
        }
        sum
    }
}

impl<A, S> ArrayBase<S, (Ix, Ix)>
    where S: Data<Elem=A>,
{
    /// Return an iterator over the elements of row `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn row_iter(&self, index: Ix) -> Elements<A, Ix>
    {
        let (m, n) = self.dim;
        let (sr, sc) = self.strides;
        assert!(index < m);
        unsafe {
            Elements { inner:
                Baseiter::new(self.ptr.offset(stride_offset(index, sr)), n, sc)
            }
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
            Elements { inner:
                Baseiter::new(self.ptr.offset(stride_offset(index, sc)), m, sr)
            }
        }
    }
}


// Matrix multiplication only defined for simple types to
// avoid trouble with failing + and *, and destructors
impl<A: Copy + linalg::Ring, S> ArrayBase<S, (Ix, Ix)>
    where S: Data<Elem=A>,
{
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
    pub fn mat_mul(&self, rhs: &ArrayBase<S, (Ix, Ix)>) -> Array<A, (Ix, Ix)>
    {
        let ((m, a), (b, n)) = (self.dim, rhs.dim);
        let (self_columns, other_rows) = (a, b);
        assert!(self_columns == other_rows);

        // Avoid initializing the memory in vec -- set it during iteration
        let mut res_elems = Vec::<A>::with_capacity(m as usize * n as usize);
        unsafe {
            res_elems.set_len(m as usize * n as usize);
        }
        let mut i = 0;
        let mut j = 0;
        for rr in res_elems.iter_mut() {
            unsafe {
                let dot = (0..a).fold(libnum::zero::<A>(),
                    |s, k| s + *self.uchk_at((i, k)) * *rhs.uchk_at((k, j))
                );
                std::ptr::write(rr, dot);
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
    pub fn mat_mul_col(&self, rhs: &ArrayBase<S, Ix>) -> Array<A, Ix>
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
                let dot = (0..a).fold(libnum::zero::<A>(),
                    |s, k| s + *self.uchk_at((i, k)) * *rhs.uchk_at(k)
                );
                std::ptr::write(rr, dot);
            }
            i += 1;
        }
        unsafe {
            ArrayBase::from_vec_dim(m, res_elems)
        }
    }
}


impl<A, S, D> ArrayBase<S, D>
    where A: Float + PartialOrd,
          S: Data<Elem=A>,
          D: Dimension
{
    /// Return `true` if the arrays' elementwise differences are all within
    /// the given absolute tolerance.<br>
    /// Return `false` otherwise, or if the shapes disagree.
    pub fn allclose<S2>(&self, rhs: &ArrayBase<S2, D>, tol: A) -> bool
        where S2: Data<Elem=A>,
    {
        self.shape() == rhs.shape() &&
        self.iter().zip(rhs.iter()).all(|(x, y)| (*x - *y).abs() <= tol)
    }
}


// Array OPERATORS

macro_rules! impl_binary_op(
    ($trt:ident, $mth:ident, $imethod:ident, $imth_scalar:ident, $doc:expr) => (
impl<A, S, D> ArrayBase<S, D> where
    A: Clone + $trt<A, Output=A>,
    S: DataMut<Elem=A>,
    D: Dimension,
{
    /// Perform elementwise
    #[doc=$doc]
    /// between `self` and `rhs`,
    /// *in place*.
    ///
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting isn't possible.
    pub fn $imethod <E: Dimension, S2> (&mut self, rhs: &ArrayBase<S2, E>)
        where S2: Data<Elem=A>,
    {
        if self.dim.ndim() == rhs.dim.ndim() &&
            self.shape() == rhs.shape() {
            for (x, y) in self.iter_mut().zip(rhs.iter()) {
                *x = (x.clone()). $mth (y.clone());
            }
        } else {
            let other_iter = rhs.broadcast_iter_unwrap(self.dim());
            for (x, y) in self.iter_mut().zip(other_iter) {
                *x = (x.clone()). $mth (y.clone());
            }
        }
    }

    /// Perform elementwise
    #[doc=$doc]
    /// between `self` and the scalar `x`,
    /// *in place*.
    pub fn $imth_scalar (&mut self, x: &A)
    {
        for elt in self.iter_mut() {
            *elt = elt.clone(). $mth (x.clone());
        }
    }
}

/// Perform elementwise
#[doc=$doc]
/// between `self` and `rhs`,
/// and return the result.
///
/// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
///
/// **Panics** if broadcasting isn't possible.
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
        if self.shape() == rhs.shape() {
            for (x, y) in self.iter_mut().zip(rhs.iter()) {
                *x = x.clone(). $mth (y.clone());
            }
        } else {
            let other_iter = rhs.broadcast_iter_unwrap(self.dim());
            for (x, y) in self.iter_mut().zip(other_iter) {
                *x = x.clone(). $mth (y.clone());
            }
        }
        self
    }
}

/// Perform elementwise
#[doc=$doc]
/// between `self` and `rhs`,
/// and return the result.
///
/// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
///
/// **Panics** if broadcasting isn't possible.
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
        let mut result = Vec::<A>::with_capacity(self.dim.size());
        if self.shape() == rhs.shape() {
            for (x, y) in self.iter().zip(rhs.iter()) {
                result.push((x.clone()). $mth (y.clone()));
            }
        } else {
            let other_iter = rhs.broadcast_iter_unwrap(self.dim());
            for (x, y) in self.iter().zip(other_iter) {
                result.push((x.clone()). $mth (y.clone()));
            }
        }
        unsafe {
            ArrayBase::from_vec_dim(self.dim.clone(), result)
        }
    }
}
    );
);

impl_binary_op!(Add, add, iadd, iadd_scalar, "Addition");
impl_binary_op!(Sub, sub, isub, isub_scalar, "Subtraction");
impl_binary_op!(Mul, mul, imul, imul_scalar, "Multiplication");
impl_binary_op!(Div, div, idiv, idiv_scalar, "Divsion");
impl_binary_op!(Rem, rem, irem, irem_scalar, "Remainder");
impl_binary_op!(BitAnd, bitand, ibitand, ibitand_scalar, "Bit and");
impl_binary_op!(BitOr, bitor, ibitor, ibitor_scalar, "Bit or");
impl_binary_op!(BitXor, bitxor, ibitxor, ibitxor_scalar, "Bit xor");
impl_binary_op!(Shl, shl, ishl, ishl_scalar, "Shift left");
impl_binary_op!(Shr, shr, ishr, ishr_scalar, "Shift right");

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
    /// **Panics** if broadcasting isn't possible.
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
            if self.shape() == rhs.shape() {
                for (x, y) in self.iter_mut().zip(rhs.iter()) {
                    x.$method(y.clone());
                }
            } else {
                let other_iter = rhs.broadcast_iter_unwrap(self.dim());
                for (x, y) in self.iter_mut().zip(other_iter) {
                    x.$method(y.clone());
                }
            }
        }
    }

        };
    }

    impl_assign_op!(AddAssign, add_assign,
                    "Implement `self += rhs` as elementwise addition (in place).\n");
    impl_assign_op!(SubAssign, sub_assign,
                    "Implement `self -= rhs` as elementwise subtraction (in place).\n");
    impl_assign_op!(MulAssign, mul_assign,
                    "Implement `self *= rhs` as elementwise multiplication (in place).\n");
    impl_assign_op!(DivAssign, div_assign,
                    "Implement `self /= rhs` as elementwise division (in place).\n");
    impl_assign_op!(RemAssign, rem_assign,
                    "Implement `self %= rhs` as elementwise remainder (in place).\n");
    impl_assign_op!(BitAndAssign, bitand_assign,
                    "Implement `self &= rhs` as elementwise bit and (in place).\n");
    impl_assign_op!(BitOrAssign, bitor_assign,
                    "Implement `self |= rhs` as elementwise bit or (in place).\n");
    impl_assign_op!(BitXorAssign, bitxor_assign,
                    "Implement `self ^= rhs` as elementwise bit xor (in place).\n");
}

impl<A, S, D> ArrayBase<S, D>
    where A: Clone + Neg<Output=A>,
          S: DataMut<Elem=A>,
          D: Dimension
{
    /// Perform an elementwise negation of `self`, *in place*.
    pub fn ineg(&mut self)
    {
        for elt in self.iter_mut() {
            *elt = elt.clone().neg()
        }
    }
}

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

impl<A, S, D> ArrayBase<S, D>
    where A: Clone + Not<Output=A>,
          S: DataMut<Elem=A>,
          D: Dimension
{
    /// Perform an elementwise unary not of `self`, *in place*.
    pub fn inot(&mut self)
    {
        for elt in self.iter_mut() {
            *elt = elt.clone().not()
        }
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

/// An iterator over the elements of an array.
///
/// Iterator element type is `&'a A`.
pub struct Elements<'a, A: 'a, D> {
    inner: Baseiter<'a, A, D>,
}

impl<'a, A, D> Elements<'a, A, D> where D: Clone
{
    /// Return the base dimension of the array being iterated.
    pub fn dim(&self) -> D
    {
        self.inner.dim.clone()
    }

    /// Return an indexed version of the iterator.
    ///
    /// Iterator element type is `(D, &'a A)`.
    ///
    /// **Note:** the indices run over the logical dimension of the iterator,
    /// i.e. a *.slice_iter()* will yield indices relative to the slice, not the
    /// base array.
    pub fn indexed(self) -> Indexed<Elements<'a, A, D>>
    {
        Indexed {
            inner: self,
        }
    }
}

/// An iterator over the elements of an array.
///
/// Iterator element type is `&'a mut A`.
pub struct ElementsMut<'a, A: 'a, D> {
    inner: Baseiter<'a, A, D>,
}

impl<'a, A, D> ElementsMut<'a, A, D> where D: Clone
{
    /// Return the base dimension of the array being iterated.
    pub fn dim(&self) -> D
    {
        self.inner.dim.clone()
    }

    /// Return an indexed version of the iterator.
    ///
    /// Iterator element type is `(D, &'a mut A)`.
    pub fn indexed(self) -> Indexed<ElementsMut<'a, A, D>>
    {
        Indexed {
            inner: self,
        }
    }
}

/// An iterator over the indexes and elements of an array.
#[derive(Clone)]
pub struct Indexed<I> {
    inner: I,
}

