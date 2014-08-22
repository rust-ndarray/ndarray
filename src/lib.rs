#![feature(macro_rules)]
#![feature(default_type_params)] /* Hash<S> */
#![crate_name="ndarray"]
#![crate_type="dylib"]

//! The **ndarray** crate provides the [**Array**](./struct.Array.html) type, an
//! n-dimensional numerical container similar to numpy's ndarray.
//!

extern crate itertools;
#[cfg(complex)]
extern crate num;

use itertools::ItertoolsClonable;
use itertools as it;

use std::fmt;
use std::hash;
use std::kinds;
use std::mem;
use std::num;
use std::default::Default;

pub use dimension::{Dimension, Shrink};

pub mod linalg;
mod dimension;

/// Array index type
pub type Ix = uint;
/// Array index type (signed)
pub type Ixs = int;

unsafe fn to_ref<A>(ptr: *const A) -> &'static A {
    mem::transmute(ptr)
}

unsafe fn to_ref_mut<A>(ptr: *mut A) -> &'static mut A {
    mem::transmute(ptr)
}

/// The **Array** type is an *N-dimensional array*.
///
/// A reference counted array with copy-on-write mutability.
///
/// The array is a container of numerical use, supporting
/// all mathematical operators by applying them elementwise.
///
/// The array is both a view and a shared owner of its data. Some methods,
/// for example [*slice()*](#method.slice), merely change the view of the data,
/// while methods like [*iadd()*](#method.iadd) allow mutating the element
/// values.
///
/// Calling a method for mutating elements, for example 
/// [*at_mut()*](#method.at_mut), [*iadd()*](#method.iadd) or
/// [*iter_mut()*](#method.iter_mut) will break sharing and require a clone of
/// the data (if it is not uniquely held).
///
/// ## Method Conventions
///
/// Methods mutating the view or array elements in place use an *i* prefix,
/// for example *slice* vs. *islice* and *add* vs *iadd*.
///
/// ## Broadcasting
///
/// Arrays support limited *broadcasting*, where arithmetic operations with
/// array operands of different sizes can be carried out by repeating the
/// elements of the smaller dimension array.
///
/// ```
/// use ndarray::arr2;
///
/// let a = arr2::<f32>([[1., 2.],
///                      [3., 4.]]);
/// let b = arr2::<f32>([[0., 1.]]);
///
/// let c = arr2::<f32>([[1., 3.],
///                      [3., 5.]]);
/// assert!(c == a + b);
/// ```
///
pub struct Array<A, D> {
    // FIXME: Unsafecell around vec needed?
    /// Rc data when used as view, Uniquely held data when being mutated
    data: std::rc::Rc<Vec<A>>,
    /// A pointer into the buffer held by data, may point anywhere
    /// in its range.
    ptr: *mut A,
    /// The size of each axis
    dim: D,
    /// The element count stride per axis. To be parsed as `int`.
    strides: D,
}

impl<A, D: Clone> Clone for Array<A, D>
{
    fn clone(&self) -> Array<A, D> {
        Array {
            data: self.data.clone(),
            ptr: self.ptr,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<A: Clone + num::Zero, D: Dimension> Array<A, D>
{
    /// Construct an Array with zeros
    pub fn zeros(dim: D) -> Array<A, D>
    {
        Array::from_elem(dim, num::zero())
    }
}

impl<A: Clone, D: Dimension> Array<A, D>
{
    /// Construct an Array with copies of `elem`
    pub fn from_elem(dim: D, elem: A) -> Array<A, D> {
        let v = Vec::from_elem(dim.size(), elem);
        unsafe {
            Array::from_vec_dim(dim, v)
        }
    }
}

impl<A> Array<A, Ix>
{
    /// Create a one-dimensional array from a vector (no allocation needed)
    pub fn from_vec(v: Vec<A>) -> Array<A, Ix> {
        unsafe {
            Array::from_vec_dim(v.len(), v)
        }
    }

    /// Create a one-dimensional array from an iterator
    pub fn from_iter<I: Iterator<A>>(mut it: I) -> Array<A, Ix> {
        Array::from_vec(it.collect())
    }

}

impl<A: Clone> Array<A, Ix>
{
    /// Create a one-dimensional array from a slice
    pub fn from_slice(xs: &[A]) -> Array<A, Ix>
    {
        Array::from_vec(xs.to_vec())
    }
}

impl<A: Clone> Array<A, (Ix, Ix)>
{
    /// Create a two-dimensional array from a slice
    ///
    /// **Fail** if the slices are not all of the same length.
    ///
    /// ```
    /// use ndarray::Array;
    /// let a = Array::from_slices([[1, 2, 3],
    ///                             [4, 5, 6i]]);
    /// assert!(a.shape() == &[2, 3]);
    /// ```
    pub fn from_slices(xs: &[&[A]]) -> Array<A, (Ix, Ix)>
    {
        unsafe {
            let (m, n) = (xs.len(), xs.get(0).map_or(0, |snd| snd.len()));
            let dim = (m, n);
            let mut result = Vec::<A>::with_capacity(dim.size());
            for &snd in xs.iter() {
                assert!(snd.len() == n);
                result.extend(snd.iter().clones())
            }
            Array::from_vec_dim(dim, result)
        }
    }
}

/// Collapse axis `axis` and shift so that only subarray `index` is
/// available.
///
/// **Fail** if `index` is larger than the size of the axis
// FIXME: Move to Dimension trait
fn do_sub<A, D: Dimension, P: Copy + RawPtr<A>>(dims: &mut D, ptr: &mut P, strides: &D,
                           axis: uint, index: uint)
{
    let dim = dims.slice()[axis];
    let stride = strides.slice()[axis] as int;
    assert!(index < dim);
    dims.slice_mut()[axis] = 1;
    let off = stride * index as int;
    unsafe {
        *ptr = ptr.offset(off);
    }
}

impl<A, D: Dimension> Array<A, D>
{
    /// Create an array from a vector (with no allocation needed).
    ///
    /// Unsafe because dimension is unchecked, and must be correct.
    pub unsafe fn from_vec_dim(dim: D, mut v: Vec<A>) -> Array<A, D> {
        Array {
            ptr: v.as_mut_ptr(),
            data: std::rc::Rc::new(v),
            strides: dim.default_strides(),
            dim: dim
        }
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

    /// Return a slice of the array's backing data in memory order.
    ///
    /// **Note:** Data memory order may not correspond to the index order
    /// of the array. Neither is the raw data slice is restricted to just the
    /// Array's view.
    pub fn raw_data<'a>(&'a self) -> &'a [A]
    {
        self.data.as_slice()
    }

    /// Return a sliced array.
    ///
    /// **Fail** if `indexes` does not match the number of array axes.
    pub fn slice(&self, indexes: &[Si]) -> Array<A, D>
    {
        let mut arr = self.clone();
        arr.islice(indexes);
        arr
    }

    /// Slice the array's view in place.
    ///
    /// **Fail** if `indexes` does not match the number of array axes.
    pub fn islice(&mut self, indexes: &[Si])
    {
        let offset = do_slices(&mut self.dim, &mut self.strides, indexes);
        unsafe {
            self.ptr = self.ptr.offset(offset);
        }
    }

    /// Return an iterator over a sliced view.
    ///
    /// **Fail** if `indexes` does not match the number of array axes.
    pub fn slice_iter<'a>(&'a self, indexes: &[Si]) -> Elements<'a, A, D>
    {
        let mut it = self.iter();
        let offset = do_slices(&mut it.inner.dim, &mut it.inner.strides, indexes);
        unsafe {
            it.inner.ptr = it.inner.ptr.offset(offset);
        }
        it
    }

    /// Return a reference to the element at `index`, or return `None`
    /// if the index is out of bounds.
    pub fn at<'a>(&'a self, index: D) -> Option<&'a A> {
        self.dim.stride_offset_checked(&self.strides, &index)
            .map(|offset| unsafe {
                to_ref(self.ptr.offset(offset) as *const _)
            })
    }

    /// Perform *unchecked* array indexing.
    ///
    /// Return a reference to the element at `index`.
    ///
    /// **Note:** only unchecked for non-debug builds of ndarray.
    #[inline]
    pub unsafe fn uchk_at<'a>(&'a self, index: D) -> &'a A {
        debug_assert!(self.dim.stride_offset_checked(&self.strides, &index).is_some());
        let off = Dimension::stride_offset(&index, &self.strides);
        to_ref(self.ptr.offset(off) as *const _)
    }

    /// Perform *unchecked* array indexing.
    ///
    /// Return a mutable reference to the element at `index`.
    ///
    /// **Note:** Only unchecked for non-debug builds of ndarray.<br>
    /// **Note:** The array must be uniquely held when mutating it.
    #[inline]
    pub unsafe fn uchk_at_mut<'a>(&'a mut self, index: D) -> &'a mut A {
        debug_assert!(std::rc::is_unique(&self.data));
        debug_assert!(self.dim.stride_offset_checked(&self.strides, &index).is_some());
        let off = Dimension::stride_offset(&index, &self.strides);
        to_ref_mut(self.ptr.offset(off))
    }

    /// Return a protoiterator
    fn base_iter<'a>(&'a self) -> Baseiter<'a, A, D>
    {
        Baseiter {
            ptr: self.ptr,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            index: Some(self.dim.first_index()),
            life: kinds::marker::ContravariantLifetime,
        }
    }

    /// Return an iterator of references to the elements of the Array
    ///
    /// Iterator element type is `&'a A`.
    pub fn iter<'a>(&'a self) -> Elements<'a, A, D>
    {
        Elements { inner: self.base_iter() }
    }

    /// Return an iterator of references to the elements of the Array
    ///
    /// Iterator element type is `(D, &'a A)`.
    pub fn indexed_iter<'a>(&'a self) -> IndexedElements<'a, A, D>
    {
        IndexedElements { inner: self.base_iter() }
    }

    /// Collapse dimension `axis` into length one,
    /// and select the subview of `index` along that axis.
    ///
    /// **Fail** if `index` is past the length of the axis.
    pub fn isubview(&mut self, axis: uint, index: uint)
    {
        do_sub(&mut self.dim, &mut self.ptr, &self.strides, axis, index)
    }

    /// Act like a larger size and/or dimension Array by *broadcasting*
    /// into a larger shape, if possible.
    ///
    /// Return `None` if shapes can not be broadcast together.
    ///
    /// ## Background
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
    /// The implementation creates an iterator with strides set to 0 for the
    /// axes that are to be repeated.
    ///
    /// See broadcasting documentation for Numpy for more information.
    pub fn broadcast_iter<'a, E: Dimension>(&'a self, dim: E)
        -> Option<Elements<'a, A, E>>
    {
        /// Return new stride when trying to grow `from` into shape `to`
        ///
        /// Broadcasting works by returning a "fake stride" where elements
        /// to repeat are in axes with 0 stride, so that several indexes point
        /// to the same element.
        ///
        /// **Note:** Cannot be used for mutable iterators, since repeating
        /// elements would create aliasing pointers.
        fn upcast<D: Dimension, E: Dimension>(to: D, from: E, stride: E) -> Option<D> {
            let mut new_stride = to.clone();
            // begin at the back (the least significant dimension)
            // size of the axis has to either agree or `from` has to be 1
            if to.ndim() < from.ndim() {
                return None
            }

            {
                let mut new_stride_iter = new_stride.slice_mut().mut_iter().rev();
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

        let broadcast_strides = 
            match upcast(dim.clone(), self.dim.clone(), self.strides.clone()) {
                Some(st) => st,
                None => return None,
            };
        let base = Baseiter {
            ptr: self.ptr,
            strides: broadcast_strides,
            index: Some(dim.first_index()),
            dim: dim,
            life: kinds::marker::ContravariantLifetime,
        };
        Some(Elements{inner: base})
    }

    /// Swap axes `ax` and `bx`.
    ///
    /// **Fail** if the axes are out of bounds.
    pub fn swap_axes(&mut self, ax: uint, bx: uint)
    {
        self.dim.slice_mut().swap(ax, bx);
        self.strides.slice_mut().swap(ax, bx);
    }

    pub fn iter1d<'b>(&'b self, axis: uint, from: &D) -> it::Stride<'b, A> {
        let dim = self.dim.slice()[axis];
        let stride = self.strides.slice()[axis];
        let off = self.dim.stride_offset_checked(&self.strides, from).unwrap();
        let ptr = unsafe {
            self.ptr.offset(off)
        };
        unsafe {
            stride_new(ptr as *const _, dim, stride as int)
        }
    }

    // Return (length, stride) for diagonal
    fn diag_params(&self) -> (uint, int)
    {
        /* empty shape has len 1 */
        let len = self.dim.slice().iter().clones().min().unwrap_or(1);
        let stride = self.strides.slice().iter()
                        .map(|x| *x as int)
                        .fold(0i, |s, a| s + a);
        return (len, stride)
    }

    /// Return an iterator over the diagonal elements of the array.
    pub fn diag_iter<'a>(&'a self) -> it::Stride<'a, A> {
        let (len, stride) = self.diag_params();
        unsafe {
            stride_new(self.ptr as *const _, len, stride)
        }
    }

    /// Return the diagonal as a one-dimensional array.
    pub fn diag(&self) -> Array<A, Ix> {
        let (len, stride) = self.diag_params();
        Array {
            data: self.data.clone(),
            ptr: self.ptr,
            dim: len,
            strides: stride as Ix,
        }
    }
}

impl<A, E: Dimension + Default, D: Dimension + Shrink<E>> Array<A, D>
{
    /// Select the subview `index` along `axis` and return a reduced
    /// dimension array.
    ///
    /// **Fail** if `index` is past the length of the axis.
    ///
    /// ```
    /// use ndarray::{arr1, arr2};
    ///
    /// let a = arr2::<f32>([[1., 2.],
    ///                      [3., 4.]]);
    /// assert_eq!(a.subview(0, 0), arr1([1., 2.]));
    /// assert_eq!(a.subview(1, 1), arr1([2., 4.]));
    /// ```
    pub fn subview(&self, axis: uint, index: uint) -> Array<A, E>
    {
        let mut res = self.clone();
        res.isubview(axis, index);
        // don't use reshape -- we always know it will fit the size,
        // and we can use from_slice on the strides as well
        Array{
            data: res.data,
            ptr: res.ptr,
            dim: res.dim.from_slice(axis),
            strides: res.strides.from_slice(axis),
        }
    }

    /*
    pub fn sub_iter<'a>(&'a self, axis: uint, index: uint) -> Elements<'a, A, E>
    {
        let mut it = self.iter();
        do_sub(&mut it.dim, &mut it.ptr, &it.strides, axis, index);
        Elements {
            ptr: it.ptr,
            dim: it.dim.from_slice(axis),
            strides: it.strides.from_slice(axis),
            index: Some(Default::default()),
            life: it.life,
        }
    }
    */
}

impl<'a, A, D: Dimension> Index<D, A> for Array<A, D>
{
    #[inline]
    fn index(&self, index: &D) -> &A {
        self.at(index.clone()).unwrap()
    }
}

impl<A: Clone, D: Dimension> Array<A, D>
{
    /// Make the Array unshared.
    ///
    /// This method is mostly only useful with unsafe code.
    pub fn ensure_unique(&mut self)
    {
        if std::rc::is_unique(&self.data) {
            return
        }
        if self.dim.size() <= self.data.len() / 2 {
            unsafe {
                *self = Array::from_vec_dim(self.dim.clone(),
                                            self.iter().clones().collect());
            }
            return;
        }
        let our_off = (self.ptr as int - self.data.as_ptr() as int)
            / mem::size_of::<A>() as int;
        let rvec = self.data.make_unique();
        unsafe {
            self.ptr = rvec.as_mut_ptr().offset(our_off);
        }
    }

    /// Return a mutable reference to the element at `index`, or return `None`
    /// if the index is out of bounds.
    pub fn at_mut<'a>(&'a mut self, index: D) -> Option<&'a mut A> {
        self.ensure_unique();
        self.dim.stride_offset_checked(&self.strides, &index)
            .map(|offset| unsafe {
                to_ref_mut(self.ptr.offset(offset))
            })
    }

    /// Return an iterator of mutable references to the elements of the Array.
    ///
    /// Iterator element type is `&'a mut A`.
    pub fn iter_mut<'a>(&'a mut self) -> ElementsMut<'a, A, D>
    {
        self.ensure_unique();
        ElementsMut { inner: self.base_iter(), nocopy: kinds::marker::NoCopy }
    }

    /// Return an iterator of indexes and mutable references to the elements of the Array.
    ///
    /// Iterator element type is `(D, &'a mut A)`.
    pub fn indexed_iter_mut<'a>(&'a mut self) -> IndexedElementsMut<'a, A, D>
    {
        self.ensure_unique();
        IndexedElementsMut { inner: self.base_iter(), nocopy: kinds::marker::NoCopy }
    }

    /// Return an iterator of mutable references into the sliced view
    /// of the array.
    ///
    /// Iterator element type is `&'a mut A`.
    ///
    /// **Fail** if `indexes` does not match the number of array axes.
    pub fn slice_iter_mut<'a>(&'a mut self, indexes: &[Si]) -> ElementsMut<'a, A, D>
    {
        let mut it = self.iter_mut();
        let offset = do_slices(&mut it.inner.dim, &mut it.inner.strides, indexes);
        unsafe {
            it.inner.ptr = it.inner.ptr.offset(offset);
        }
        it
    }


    /// Select the subview `index` along `axis` and return an iterator
    /// of the subview.
    ///
    /// Iterator element type is `&'a mut A`.
    ///
    /// **Fail** if `axis` or `index` is out of bounds.
    pub fn sub_iter_mut<'a>(&'a mut self, axis: uint, index: uint)
        -> ElementsMut<'a, A, D>
    {
        let mut it = self.iter_mut();
        do_sub(&mut it.inner.dim, &mut it.inner.ptr, &it.inner.strides, axis, index);
        it
    }

    /// Return an iterator over the diagonal elements of the array.
    pub fn diag_iter_mut<'a>(&'a mut self) -> it::StrideMut<'a, A> {
        self.ensure_unique();
        let (len, stride) = self.diag_params();
        unsafe {
            stride_mut(self.ptr, len, stride)
        }
    }

    /// Return a mutable slice of the array's backing data in memory order.
    ///
    /// **Note:** Data memory order may not correspond to the index order
    /// of the array. Neither is the raw data slice is restricted to just the
    /// Array's view.
    ///
    /// **Note:** The data is uniquely held and nonaliased
    /// while it is mutably borrowed.
    pub fn raw_data_mut<'a>(&'a mut self) -> &'a mut [A]
    {
        self.data.make_unique().as_mut_slice()
    }


    /// Transform the array into `shape`; any other shape
    /// with the same number of elements is accepted.
    ///
    /// **Fail** if sizes are incompatible.
    pub fn reshape<E: Dimension>(&self, shape: E) -> Array<A, E> {
        if shape.size() != self.dim.size() {
            fail!("Incompatible sizes in reshape, attempted from: {}, to: {}",
                  self.dim.slice(), shape.slice())
        }
        // Check if contiguous, if not => copy all, else just adapt strides
        if self.is_standard_layout() {
            let cl = self.clone();
            Array{
                data: cl.data,
                ptr: cl.ptr,
                strides: shape.default_strides(),
                dim: shape,
            }
        } else {
            let v = self.iter().map(|x| x.clone()).collect::<Vec<A>>();
            unsafe {
                Array::from_vec_dim(shape, v)
            }
        }
    }

    /// Perform an elementwise assigment to `self` from `other`.
    ///
    /// If their shapes disagree, `other` is broadcast to the shape of `self`.
    ///
    /// **Fail** if broadcasting isn't possible.
    pub fn assign<E: Dimension>(&mut self, other: &Array<A, E>)
    {
        if self.shape() == other.shape() {
            for (x, y) in self.iter_mut().zip(other.iter()) {
                *x = y.clone();
            }
        } else {
            let other_iter = match other.broadcast_iter(self.dim()) {
                Some(it) => it,
                None => fail!("{}: Could not broadcast array from shape {} into: {}",
                              "assign", other.shape(), self.shape())
            };
            for (x, y) in self.iter_mut().zip(other_iter) {
                *x = y.clone();
            }
        }
    }

    /// Perform an elementwise assigment to `self` from scalar `x`.
    pub fn assign_scalar(&mut self, x: &A)
    {
        for elt in self.iter_mut() {
            *elt = x.clone();
        }
    }
}

impl<'a, A: Clone, D: Dimension> IndexMut<D, A> for Array<A, D>
{
    #[inline]
    fn index_mut(&mut self, index: &D) -> &mut A {
        self.at_mut(index.clone()).unwrap()
    }
}

unsafe fn stride_new<A>(ptr: *const A, len: uint, stride: int) -> it::Stride<'static, A>
{
    it::Stride::from_ptr_len(ptr, len, stride)
}

// NOTE: lifetime
unsafe fn stride_mut<A>(ptr: *mut A, len: uint, stride: int) -> it::StrideMut<'static, A>
{
    it::StrideMut::from_ptr_len(ptr, len, stride)
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
    Array::from_slice(xs)
}

/// Return a two-dimensional array with elements from `xs`.
///
/// **Fail** if the slices are not all of the same length.
pub fn arr2<A: Clone>(xs: &[&[A]]) -> Array<A, (Ix, Ix)>
{
    Array::from_slices(xs)
}

impl<A: Clone + linalg::Field,
     E: Dimension + Default, D: Dimension + Shrink<E>>
    Array<A, D>
{
    /// Return sum along `axis`.
    ///
    /// ```
    /// use ndarray::{arr0, arr1, arr2};
    ///
    /// let a = arr2::<f32>([[1., 2.],
    ///                      [3., 4.]]);
    /// assert_eq!(a.sum(0), arr1([4., 6.]));
    /// assert_eq!(a.sum(1), arr1([3., 7.]));
    ///
    /// assert_eq!(a.sum(0).sum(0), arr0(10.));
    /// ```
    ///
    /// **Fail** if `axis` is out of bounds.
    pub fn sum(&self, axis: uint) -> Array<A, E>
    {
        let n = self.shape()[axis];
        let mut res = self.subview(axis, 0);
        for i in range(1, n) {
            res.iadd(&self.subview(axis, i))
        }
        res
    }

    /// Return mean along `axis`.
    ///
    /// **Fail** if `axis` is out of bounds.
    pub fn mean(&self, axis: uint) -> Array<A, E>
    {
        let n = self.shape()[axis];
        let mut sum = self.sum(axis);
        let one = num::one::<A>();
        let mut cnt = one.clone();
        for i in range(1, n) {
            cnt = cnt + one;
        }
        for elt in sum.iter_mut() {
            *elt = *elt / cnt;
        }
        sum
    }
}

impl<A> Array<A, (Ix, Ix)>
{
    /// Return an iterator over the elements of row `index`.
    ///
    /// **Fail** if `index` is out of bounds.
    pub fn row_iter<'a>(&'a self, index: uint) -> it::Stride<'a, A>
    {
        let (m, n) = self.dim;
        let (sr, sc) = self.strides;
        let (sr, sc) = (sr as int, sc as int);
        assert!(index < m);
        unsafe {
            stride_new(self.ptr.offset(sr * index as int) as *const A, n, sc)
        }
    }

    /// Return an iterator over the elements of column `index`.
    ///
    /// **Fail** if `index` is out of bounds.
    pub fn col_iter<'a>(&'a self, index: uint) -> it::Stride<'a, A>
    {
        let (m, n) = self.dim;
        let (sr, sc) = self.strides;
        let (sr, sc) = (sr as int, sc as int);
        assert!(index < n);
        unsafe {
            stride_new(self.ptr.offset(sc * index as int) as *const A, m, sr)
        }
    }
}


// Matrix multiplication only defined for simple types to
// avoid trouble with failing + and *, and destructors
impl<'a, A: Copy + linalg::Ring> Array<A, (Ix, Ix)>
{
    /// Perform matrix multiplication of rectangular arrays `self` and `other`.
    ///
    /// The array sizes must agree in the way that
    /// if `self` is *M* × *N*, then `other` is *N* × *K*.
    ///
    /// Return a result array with shape *M* × *K*.
    ///
    /// **Fail** if sizes are incompatible.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2::<f32>([[1., 2.],
    ///                      [0., 1.]]);
    /// let b = arr2::<f32>([[1., 2.],
    ///                      [2., 3.]]);
    /// assert_eq!(a.mat_mul(&b), arr2([[5., 8.],
    ///                                 [2., 3.]]));
    /// ```
    ///
    pub fn mat_mul(&self, other: &Array<A, (Ix, Ix)>) -> Array<A, (Ix, Ix)>
    {
        let ((m, a), (b, n)) = (self.dim, other.dim);
        let (self_columns, other_rows) = (a, b);
        assert!(self_columns == other_rows);

        // Avoid initializing the memory in vec -- set it during iteration
        let mut res_elems = Vec::<A>::with_capacity(m * n);
        unsafe {
            res_elems.set_len(m * n);
        }
        let mut i = 0;
        let mut j = 0;
        for rr in res_elems.mut_iter() {
            unsafe {
                let dot = range(0, a).fold(num::zero::<A>(),
                    |s, k| s + *self.uchk_at((i, k)) * *other.uchk_at((k, j))
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
            Array::from_vec_dim((m, n), res_elems)
        }
    }

    /// Perform the matrix multiplication of the rectangular array `self` and
    /// column vector `other`.
    ///
    /// The array sizes must agree in the way that
    /// if `self` is *M* × *N*, then `other` is *N*.
    ///
    /// Return a result array with shape *M*.
    ///
    /// **Fail** if sizes are incompatible.
    pub fn mat_mul_col(&self, other: &Array<A, Ix>) -> Array<A, Ix>
    {
        let ((m, a), n) = (self.dim, other.dim);
        let (self_columns, other_rows) = (a, n);
        assert!(self_columns == other_rows);

        // Avoid initializing the memory in vec -- set it during iteration
        let mut res_elems = Vec::<A>::with_capacity(m);
        unsafe {
            res_elems.set_len(m);
        }
        let mut i = 0;
        for rr in res_elems.mut_iter() {
            unsafe {
                let dot = range(0, a).fold(num::zero::<A>(),
                    |s, k| s + *self.uchk_at((i, k)) * *other.uchk_at(k)
                );
                std::ptr::write(rr, dot);
            }
            i += 1;
        }
        unsafe {
            Array::from_vec_dim(m, res_elems)
        }
    }
}



fn format_array<A, D: Dimension>(view: &Array<A, D>, f: &mut fmt::Formatter,
                                 format: |&mut fmt::Formatter, &A| -> fmt::Result)
                                -> fmt::Result
{
    let sz = view.dim.slice().len();
    if sz > 0 && f.width.is_none() {
        f.width = Some(4)
    }
    let mut last_index = view.dim.first_index();
    for _ in range(0, sz) {
        try!(write!(f, "["));
    }
    let mut first = true;
    // Simply use the indexed iterator, and take the index wraparounds
    // as cues for when to add []'s and how many to add.
    for (index, elt) in view.indexed_iter() {
        let mut update_index = false;
        for (i, (a, b)) in index.slice().iter().take(sz-1)
                        .zip(last_index.slice().iter())
                        .enumerate()
        {
            if a != b {
                // New row.
                // # of ['s needed
                let n = sz - i - 1;
                for _ in range(0, n) {
                    try!(write!(f, "]"));
                }
                try!(write!(f, ",\n"));
                for _ in range(0, sz - n) {
                    try!(write!(f, " "));
                }
                for _ in range(0, n) {
                    try!(write!(f, "["));
                }
                first = true;
                update_index = true;
                break;
            }
        }
        if !first {
            try!(write!(f, ", "));
        }
        first = false;
        try!(format(f, elt));

        if update_index {
            last_index = index;
        }
    }
    for _ in range(0, sz) {
        try!(write!(f, "]"));
    }
    Ok(())
}

// NOTE: We can impl other fmt traits here
impl<'a, A: fmt::Show, D: Dimension> fmt::Show for Array<A, D>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        format_array(self, f, |f, elt| elt.fmt(f))
    }
}

// Array OPERATORS

impl<A: PartialEq, D: Dimension>
PartialEq for Array<A, D>
{
    /// Return `true` if all elements of `self` and `other` are equal.
    ///
    /// **Fail** if shapes are not equal.
    fn eq(&self, other: &Array<A, D>) -> bool
    {
        assert!(self.shape() == other.shape());
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<A: Eq, D: Dimension>
Eq for Array<A, D> {}

macro_rules! impl_binary_op(
    ($trt:ident, $mth:ident, $imethod:ident, $imth_scalar:ident) => (
impl<A: Clone + $trt<A, A>, D: Dimension>
Array<A, D>
{
    /// Perform an elementwise arithmetic operation between `self` and `other`,
    /// *in place*.
    ///
    /// If their shapes disagree, `other` is broadcast to the shape of `self`.
    ///
    /// **Fail** if broadcasting isn't possible.
    pub fn $imethod <E: Dimension> (&mut self, other: &Array<A, E>)
    {
        if self.dim.ndim() == other.dim.ndim() &&
            self.shape() == other.shape() {
            for (x, y) in self.iter_mut().zip(other.iter()) {
                *x = (*x). $mth (y);
            }
        } else {
            let other_iter = match other.broadcast_iter(self.dim()) {
                Some(it) => it,
                None => fail!("{}: Could not broadcast array from shape {} into: {}",
                              stringify!($imethod), other.shape(), self.shape())
            };
            for (x, y) in self.iter_mut().zip(other_iter) {
                *x = (*x). $mth (y);
            }
        }
    }

    /// Perform an elementwise arithmetic operation between `self` and the scalar `x`,
    /// *in place*.
    pub fn $imth_scalar (&mut self, x: &A)
    {
        for elt in self.iter_mut() {
            *elt = elt. $mth (x);
        }
    }
}

impl<A: $trt<A, A>, D: Dimension, E: Dimension>
$trt<Array<A, E>, Array<A, D>> for Array<A, D>
{
    /// Perform an elementwise arithmetic operation between `self` and `other`,
    /// and return the result.
    ///
    /// If their shapes disagree, `other` is broadcast to the shape of `self`.
    ///
    /// **Fail** if broadcasting isn't possible.
    fn $mth (&self, other: &Array<A, E>) -> Array<A, D>
    {
        // FIXME: Can we co-broadcast arrays here? And how?
        let mut result = Vec::<A>::with_capacity(self.dim.size());
        if self.shape() == other.shape() {
            for (x, y) in self.iter().zip(other.iter()) {
                result.push((*x). $mth (y));
            }
        } else {
            let other_iter = match other.broadcast_iter(self.dim()) {
                Some(it) => it,
                None => fail!("{}: Could not broadcast array from shape {} into: {}",
                              stringify!($mth), other.shape(), self.shape())
            };
            for (x, y) in self.iter().zip(other_iter) {
                result.push((*x). $mth (y));
            }
        }
        unsafe {
            Array::from_vec_dim(self.dim.clone(), result)
        }
    }
}
    );
)

impl_binary_op!(Add, add, iadd, iadd_scalar)
impl_binary_op!(Sub, sub, isub, isub_scalar)
impl_binary_op!(Mul, mul, imul, imul_scalar)
impl_binary_op!(Div, div, idiv, idiv_scalar)
impl_binary_op!(Rem, rem, irem, irem_scalar)
impl_binary_op!(BitAnd, bitand, ibitand, ibitand_scalar)
impl_binary_op!(BitOr, bitor, ibitor, ibitor_scalar)
impl_binary_op!(BitXor, bitxor, ibitxor, ibitxor_scalar)
impl_binary_op!(Shl, shl, ishl, ishl_scalar)
impl_binary_op!(Shr, shr, ishr, ishr_scalar)

impl<A: Clone + Neg<A>, D: Dimension>
Array<A, D>
{
    /// Perform an elementwise negation of `self`, *in place*.
    pub fn ineg(&mut self)
    {
        for elt in self.iter_mut() {
            *elt = (*elt).neg()
        }
    }
}

impl<A: Clone + Neg<A>, D: Dimension>
Neg<Array<A, D>> for Array<A, D>
{
    /// Perform an elementwise negation of `self` and return the result.
    fn neg(&self) -> Array<A, D>
    {
        let mut res = self.clone();
        res.ineg();
        res
    }
}

impl<A: Clone + Not<A>, D: Dimension>
Array<A, D>
{
    /// Perform an elementwise unary not of `self`, *in place*.
    pub fn inot(&mut self)
    {
        for elt in self.iter_mut() {
            *elt = (*elt).not()
        }
    }
}

impl<A: Clone + Not<A>, D: Dimension>
Not<Array<A, D>> for Array<A, D>
{
    /// Perform an elementwise unary not of `self` and return the result.
    fn not(&self) -> Array<A, D>
    {
        let mut res = self.clone();
        res.inot();
        res
    }
}

/// Base for array iterators
///
/// Iterator element type is `&'a A`.
struct Baseiter<'a, A, D> {
    ptr: *mut A,
    dim: D,
    strides: D,
    index: Option<D>,
    life: kinds::marker::ContravariantLifetime<'a>,
}

impl<'a, A, D: Dimension> Baseiter<'a, A, D>
{
    #[inline]
    fn next(&mut self) -> Option<*mut A>
    {
        let index = match self.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        let offset = Dimension::stride_offset(&index, &self.strides);
        self.index = self.dim.next_for(index);
        unsafe {
            Some(self.ptr.offset(offset))
        }
    }

    fn size_hint(&self) -> uint
    {
        match self.index {
            None => 0,
            Some(ref ix) => {
                let gone = self.dim.default_strides().slice().iter()
                            .zip(ix.slice().iter())
                                 .fold(0u, |s, (&a, &b)| s + a * b);
                self.dim.size() - gone
            }
        }
    }
}

impl<'a, A> Baseiter<'a, A, Ix>
{
    #[inline]
    fn next_back(&mut self) -> Option<*mut A>
    {
        let index = match self.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        self.dim -= 1;
        let offset = Dimension::stride_offset(&self.dim, &self.strides);
        if index == self.dim {
            self.index = None;
        }

        unsafe {
            Some(self.ptr.offset(offset))
        }
    }
}

impl<'a, A, D: Clone> Clone for Baseiter<'a, A, D>
{
    fn clone(&self) -> Baseiter<'a, A, D>
    {
        Baseiter {
            ptr: self.ptr,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            index: self.index.clone(),
            life: self.life
        }
    }
}

/// An iterator over the elements of an array.
///
/// Iterator element type is `&'a A`.
pub struct Elements<'a, A, D> {
    inner: Baseiter<'a, A, D>,
}

impl<'a, A, D: Clone> Clone for Elements<'a, A, D>
{
    fn clone(&self) -> Elements<'a, A, D> { Elements{inner: self.inner.clone()} }
}

impl<'a, A, D: Dimension> Iterator<&'a A> for Elements<'a, A, D>
{
    #[inline]
    fn next(&mut self) -> Option<&'a A>
    {
        unsafe {
            self.inner.next().map(|p| to_ref(p as *const _))
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>)
    {
        let len = self.inner.size_hint();
        (len, Some(len))
    }
}

impl<'a, A> DoubleEndedIterator<&'a A> for Elements<'a, A, Ix>
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a A>
    {
        unsafe {
            self.inner.next_back().map(|p| to_ref(p as *const _))
        }
    }
}

impl<'a, A> ExactSize<&'a A> for Elements<'a, A, Ix> { }

/// An iterator over the indexes and elements of an array.
///
/// Iterator element type is `(D, &'a A)`.
pub struct IndexedElements<'a, A, D> {
    inner: Baseiter<'a, A, D>,
}

impl<'a, A, D: Clone> Clone for IndexedElements<'a, A, D>
{
    fn clone(&self) -> IndexedElements<'a, A, D> {
        IndexedElements{inner: self.inner.clone()}
    }
}

impl<'a, A, D: Dimension> Iterator<(D, &'a A)> for IndexedElements<'a, A, D>
{
    #[inline]
    fn next(&mut self) -> Option<(D, &'a A)>
    {
        let index = match self.inner.index {
            None => return None,
            Some(ref ix) => ix.clone()
        };
        unsafe {
            match self.inner.next() {
                None => None,
                Some(p) => Some((index, to_ref(p as *const _)))
            }
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>)
    {
        let len = self.inner.size_hint();
        (len, Some(len))
    }
}

/// An iterator over the elements of an array.
///
/// Iterator element type is `&'a mut A`.
pub struct ElementsMut<'a, A, D> {
    inner: Baseiter<'a, A, D>,
    nocopy: kinds::marker::NoCopy,
}

impl<'a, A, D: Dimension> Iterator<&'a mut A> for ElementsMut<'a, A, D>
{
    #[inline]
    fn next(&mut self) -> Option<&'a mut A>
    {
        unsafe {
            self.inner.next().map(|p| to_ref_mut(p))
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>)
    {
        let len = self.inner.size_hint();
        (len, Some(len))
    }
}

impl<'a, A> DoubleEndedIterator<&'a mut A> for ElementsMut<'a, A, Ix>
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut A>
    {
        unsafe {
            self.inner.next_back().map(|p| to_ref_mut(p))
        }
    }
}

/// An iterator over the indexes and elements of an array.
///
/// Iterator element type is `(D, &'a mut A)`.
pub struct IndexedElementsMut<'a, A, D> {
    inner: Baseiter<'a, A, D>,
    nocopy: kinds::marker::NoCopy,
}

impl<'a, A, D: Dimension> Iterator<(D, &'a mut A)> for IndexedElementsMut<'a, A, D>
{
    #[inline]
    fn next(&mut self) -> Option<(D, &'a mut A)>
    {
        let index = match self.inner.index {
            None => return None,
            Some(ref ix) => ix.clone()
        };
        unsafe {
            match self.inner.next() {
                None => None,
                Some(p) => Some((index, to_ref_mut(p)))
            }
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>)
    {
        let len = self.inner.size_hint();
        (len, Some(len))
    }
}

/// An iterator of the indexes of an array shape.
///
/// Iterator element type is `D`.
#[deriving(Clone)]
pub struct Indexes<D> {
    dim: D,
    index: Option<D>,
}

impl<D: Dimension> Indexes<D>
{
    /// Create an iterator over the array shape `dim`.
    pub fn new(dim: D) -> Indexes<D>
    {
        Indexes {
            index: Some(dim.first_index()),
            dim: dim,
        }
    }
}


impl<D: Dimension> Iterator<D> for Indexes<D>
{
    #[inline]
    fn next(&mut self) -> Option<D>
    {
        let index = match self.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        self.index = self.dim.next_for(index.clone());
        Some(index)
    }

    fn size_hint(&self) -> (uint, Option<uint>)
    {
        let l = match self.index {
            None => 0,
            Some(ref ix) => {
                let gone = self.dim.default_strides().slice().iter()
                            .zip(ix.slice().iter())
                                 .fold(0u, |s, (&a, &b)| s + a * b);
                self.dim.size() - gone
            }
        };
        (l, Some(l))
    }
}

// [a:b:s] syntax for example [:3], [::-1]
// [0,:] -- first row of matrix
// [:,0] -- first column of matrix

#[deriving(Clone, PartialEq, Eq, Hash, Show)]
/// A slice, a description of a range of an array axis.
///
/// Fields are `begin`, `end` and `stride`, where
/// negative `begin` or `end` indexes are counted from the back
/// of the axis.
///
/// If `end` is `None`, the slice extends to the end of the axis.
///
/// ## Examples
///
/// `Si(0, None, 1)` is the full range of an axis.
/// Python equivalent is `[:]`.
///
/// `Si(a, Some(b), 2)` is every second element from `a` until `b`.
/// Python equivalent is `[a:b:2]`.
///
/// `Si(a, None, -1)` is every element, in reverse order, from `a`
/// until the end. Python equivalent is `[a::-1]`
pub struct Si(pub Ixs, pub Option<Ixs>, pub Ixs);

/// Slice value for the full range of an axis.
pub static S: Si = Si(0, None, 1);

fn abs_index(len: Ixs, index: Ixs) -> Ix {
    if index < 0 {
        (len + index) as Ix
    } else { index as Ix }
}

/// Modify dimension, strides and return data pointer offset
// FIXME: Move to Dimension trait
fn do_slices<D: Dimension>(dim: &mut D, strides: &mut D, slices: &[Si]) -> int
{
    let mut offset = 0;
    assert!(slices.len() == dim.slice().len());
    for ((dr, sr), &slc) in dim.slice_mut().mut_iter()
                            .zip(strides.slice_mut().mut_iter())
                            .zip(slices.iter())
    {
        let m = *dr;
        let mi = m as int;
        let Si(b1, opt_e1, s1) = slc;
        let e1 = opt_e1.unwrap_or(mi);

        let b1 = abs_index(mi, b1);
        let mut e1 = abs_index(mi, e1);
        if e1 < b1 { e1 = b1; }

        assert!(b1 <= m);
        assert!(e1 <= m);

        let m = e1 - b1;
        // stride
        let s = (*sr) as int;

        // Data pointer offset
        offset += b1 as int * s;
        // Adjust for strides
        assert!(s1 != 0);
        // How to implement negative strides:
        //
        // Increase start pointer by
        // old stride * (old dim - 1)
        // to put the pointer completely in the other end
        if s1 < 0 {
            offset += s * ((m - 1) as int);
        }

        let s_prim = s * s1;

        let (d, r) = num::div_rem(m, s1.abs() as uint);
        let m_prim = d + if r > 0 { 1 } else { 0 };

        // Update dimension and stride coordinate
        *dr = m_prim;
        *sr = s_prim as uint;
    }
    offset
}

impl<A> FromIterator<A> for Array<A, Ix>
{
    fn from_iter<I: Iterator<A>>(it: I) -> Array<A, Ix>
    {
        Array::from_iter(it)
    }
}

impl<S: hash::Writer, A: hash::Hash<S>, D: Dimension>
hash::Hash<S> for Array<A, D>
{
    fn hash(&self, state: &mut S)
    {
        self.shape().hash(state);
        for elt in self.iter() {
            elt.hash(state)
        }
    }
}
