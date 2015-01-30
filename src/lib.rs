#![feature(
    core,
    hash,
    alloc,
    rand,
    )]
#![crate_name="ndarray"]
#![crate_type="dylib"]

//! The **ndarray** crate provides the [**Array**](./struct.Array.html) type, an
//! n-dimensional container similar to numpy's ndarray.
//!

#[cfg(not(nocomplex))]
extern crate "num" as libnum;
extern crate "rustc-serialize" as serialize;

use std::mem;
use std::num::Float;
use std::ops::{Add, Sub, Mul, Div, Rem, Neg, Not, Shr, Shl,
    BitAnd,
    BitOr,
    BitXor,
};
use std::rand::{Rand, Rng};

pub use dimension::{Dimension, RemoveAxis, Si, S};
use dimension::stride_offset;

pub use indexes::Indexes;
pub use indexes::ixrange;

use iterators::Baseiter;

pub mod linalg;
mod arraytraits;
mod arrayformat;
mod dimension;
mod indexes;
mod iterators;
//mod macros;

// NOTE: In theory, the whole library should compile
// and pass tests even if you change Ix and Ixs.
/// Array index type
pub type Ix = u32;
/// Array index type (signed)
pub type Ixs = i32;

unsafe fn to_ref<'a, A>(ptr: *const A) -> &'a A {
    mem::transmute(ptr)
}

unsafe fn to_ref_mut<'a, A>(ptr: *mut A) -> &'a mut A {
    mem::transmute(ptr)
}

/// The **Array** type is an *N-dimensional array*.
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
/// [*at_mut()*](#method.at_mut), [*iadd()*](#method.iadd) or
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
/// Arrays use **u32** for indexing, represented by the types **Ix** and **Ixs** 
/// (signed).
///
/// ## Broadcasting
///
/// Arrays support limited *broadcasting*, where arithmetic operations with
/// array operands of different sizes can be carried out by repeating the
/// elements of the smaller dimension array. See
/// [*.broadcast_iter()*](#method.broadcast_iter) for a more detailed
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
pub struct Array<A, D> {
    // FIXME: Unsafecell around vec needed?
    /// Rc data when used as view, Uniquely held data when being mutated
    data: std::rc::Rc<Vec<A>>,
    /// A pointer into the buffer held by data, may point anywhere
    /// in its range.
    ptr: *mut A,
    /// The size of each axis
    dim: D,
    /// The element count stride per axis. To be parsed as **isize**.
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

impl<A> Array<A, Ix>
{
    /// Create a one-dimensional array from a vector (no allocation needed).
    pub fn from_vec(v: Vec<A>) -> Array<A, Ix> {
        unsafe {
            Array::from_vec_dim(v.len() as Ix, v)
        }
    }

    /// Create a one-dimensional array from an iterator.
    pub fn from_iter<I: Iterator<Item=A>>(it: I) -> Array<A, Ix> {
        Array::from_vec(it.collect())
    }
}

impl Array<f32, Ix>
{
    /// Create a one-dimensional Array from interval **[begin, end)**
    pub fn range(begin: f32, end: f32) -> Array<f32, Ix>
    {
        Array::from_iter(std::iter::count(begin, 1.0).take_while(|&x| x < end))
    }
}

impl<A, D> Array<A, D> where D: Dimension
{
    /// Construct an Array with zeros.
    pub fn zeros(dim: D) -> Array<A, D> where A: Clone + libnum::Zero
    {
        Array::from_elem(dim, libnum::zero())
    }

    /// Construct an Array with copies of **elem**.
    ///
    /// ## Example
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
    pub fn from_elem(dim: D, elem: A) -> Array<A, D> where A: Clone
    {
        let v = std::iter::repeat(elem).take(dim.size()).collect();
        unsafe {
            Array::from_vec_dim(dim, v)
        }
    }

    /// Create an array from a vector (with no allocation needed).
    ///
    /// Unsafe because dimension is unchecked, and must be correct.
    pub unsafe fn from_vec_dim(dim: D, mut v: Vec<A>) -> Array<A, D>
    {
        debug_assert!(dim.size() == v.len());
        Array {
            ptr: v.as_mut_ptr(),
            data: std::rc::Rc::new(v),
            strides: dim.default_strides(),
            dim: dim
        }
    }

    /// Construct an Array with random elements
    pub fn random<R: Rng>(dim: D, rng: &mut R) -> Array<A, D> where A: Rand
    {
        unsafe {
            let sz = dim.size();
            Array::from_vec_dim(dim, rng.gen_iter().take(sz).collect())
        }
    }

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

    /// Return **true** if the array data is laid out in
    /// contiguous “C order” where the last index is the most rapidly
    /// varying.
    ///
    /// Return **false** otherwise, i.e the array is possibly not
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
        &(*self.data)[]
    }

    /// Return a sliced array.
    ///
    /// **Panics** if **indexes** does not match the number of array axes.
    pub fn slice(&self, indexes: &[Si]) -> Array<A, D>
    {
        let mut arr = self.clone();
        arr.islice(indexes);
        arr
    }

    /// Slice the array's view in place.
    ///
    /// **Panics** if **indexes** does not match the number of array axes.
    pub fn islice(&mut self, indexes: &[Si])
    {
        let offset = Dimension::do_slices(&mut self.dim, &mut self.strides, indexes);
        unsafe {
            self.ptr = self.ptr.offset(offset);
        }
    }

    /// Return an iterator over a sliced view.
    ///
    /// **Panics** if **indexes** does not match the number of array axes.
    pub fn slice_iter<'a>(&'a self, indexes: &[Si]) -> Elements<'a, A, D>
    {
        let mut it = self.iter();
        let offset = Dimension::do_slices(&mut it.inner.dim, &mut it.inner.strides, indexes);
        unsafe {
            it.inner.ptr = it.inner.ptr.offset(offset);
        }
        it
    }

    /// Return a reference to the element at **index**, or return **None** 
    /// if the index is out of bounds.
    pub fn at<'a>(&'a self, index: D) -> Option<&'a A> {
        self.dim.stride_offset_checked(&self.strides, &index)
            .map(|offset| unsafe {
                to_ref(self.ptr.offset(offset) as *const _)
            })
    }

    /// Perform *unchecked* array indexing.
    ///
    /// Return a reference to the element at **index**.
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
    /// Return a mutable reference to the element at **index**.
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
    #[inline]
    fn base_iter<'a>(&'a self) -> Baseiter<'a, A, D>
    {
        unsafe {
            Baseiter::new(self.ptr, self.dim.clone(), self.strides.clone())
        }
    }

    /// Return an iterator of references to the elements of the array.
    ///
    /// Iterator element type is **&'a A**.
    pub fn iter<'a>(&'a self) -> Elements<'a, A, D>
    {
        Elements { inner: self.base_iter() }
    }

    /// Return an iterator of references to the elements of the array.
    ///
    /// Iterator element type is **(D, &'a A)**.
    pub fn indexed_iter<'a>(&'a self) -> IndexedElements<'a, A, D>
    {
        IndexedElements { inner: self.base_iter() }
    }

    /// Collapse dimension **axis** into length one,
    /// and select the subview of **index** along that axis.
    ///
    /// **Panics** if **index** is past the length of the axis.
    pub fn isubview(&mut self, axis: usize, index: Ix)
    {
        dimension::do_sub(&mut self.dim, &mut self.ptr, &self.strides, axis, index)
    }

    /// Act like a larger size and/or shape array by *broadcasting*
    /// into a larger shape, if possible.
    ///
    /// Return **None** if shapes can not be broadcast together.
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
    ///
    /// ## Example
    ///
    /// ```
    /// use ndarray::arr1;
    ///
    /// assert!(
    ///     arr1(&[1., 0.]).broadcast_iter((10, 2)).unwrap().count()
    ///     == 20
    /// );
    /// ```
    pub fn broadcast_iter<'a, E: Dimension>(&'a self, dim: E)
        -> Option<Elements<'a, A, E>>
    {
        /// Return new stride when trying to grow **from** into shape **to**
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

        let broadcast_strides = 
            match upcast(&dim, &self.dim, &self.strides) {
                Some(st) => st,
                None => return None,
            };
        Some(Elements {
            inner:
            unsafe {
                Baseiter::new(self.ptr, dim, broadcast_strides)
            }
        })
    }

    #[inline(never)]
    fn broadcast_iter_unwrap<'a, E: Dimension>(&'a self, dim: E)
        -> Elements<'a, A, E>
    {
        match self.broadcast_iter(dim.clone()) {
            Some(it) => it,
            None => panic!("Could not broadcast array from shape {:?} into: {:?}",
                           self.shape(), dim.slice())
        }
    }

    /// Swap axes **ax** and **bx**.
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
    pub fn diag_iter<'a>(&'a self) -> Elements<'a, A, Ix>
    {
        let (len, stride) = self.diag_params();
        unsafe {
            Elements { inner:
                Baseiter::new(self.ptr, len, stride as Ix)
            }
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

    /// Apply **f** elementwise and return a new array with
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
    pub fn map<'a, B, F>(&'a self, mut f: F) -> Array<B, D> where
        F: FnMut(&'a A) -> B
    {
        let mut res = Vec::<B>::with_capacity(self.dim.size());
        for elt in self.iter() {
            res.push(f(elt))
        }
        unsafe {
            Array::from_vec_dim(self.dim.clone(), res)
        }
    }

    /// Select the subview **index** along **axis** and return an
    /// array with that axis removed.
    ///
    /// **Panics** if **index** is past the length of the axis.
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
    pub fn subview(&self, axis: usize, index: Ix) -> Array<A, <D as RemoveAxis>::Smaller> where
        D: RemoveAxis
    {
        let mut res = self.clone();
        res.isubview(axis, index);
        // don't use reshape -- we always know it will fit the size,
        // and we can use remove_axis on the strides as well
        Array{
            data: res.data,
            ptr: res.ptr,
            dim: res.dim.remove_axis(axis),
            strides: res.strides.remove_axis(axis),
        }
    }

    /// Make the array unshared.
    ///
    /// This method is mostly only useful with unsafe code.
    pub fn ensure_unique(&mut self) where A: Clone
    {
        if std::rc::is_unique(&self.data) {
            return
        }
        if self.dim.size() <= self.data.len() / 2 {
            unsafe {
                *self = Array::from_vec_dim(self.dim.clone(),
                                            self.iter().map(|x| x.clone()).collect());
            }
            return;
        }
        let our_off = (self.ptr as isize - self.data.as_ptr() as isize)
            / mem::size_of::<A>() as isize;
        let rvec = self.data.make_unique();
        unsafe {
            self.ptr = rvec.as_mut_ptr().offset(our_off);
        }
    }

    /// Return a mutable reference to the element at **index**, or return **None**
    /// if the index is out of bounds.
    pub fn at_mut<'a>(&'a mut self, index: D) -> Option<&'a mut A> where A: Clone
    {
        self.ensure_unique();
        self.dim.stride_offset_checked(&self.strides, &index)
            .map(|offset| unsafe {
                to_ref_mut(self.ptr.offset(offset))
            })
    }

    /// Return an iterator of mutable references to the elements of the array.
    ///
    /// Iterator element type is **&'a mut A**.
    pub fn iter_mut<'a>(&'a mut self) -> ElementsMut<'a, A, D> where A: Clone
    {
        self.ensure_unique();
        ElementsMut { inner: self.base_iter() }
    }

    /// Return an iterator of indexes and mutable references to the elements of the array.
    ///
    /// Iterator element type is **(D, &'a mut A)**.
    pub fn indexed_iter_mut<'a>(&'a mut self) -> IndexedElementsMut<'a, A, D> where A: Clone
    {
        self.ensure_unique();
        IndexedElementsMut { inner: self.base_iter() }
    }

    /// Return an iterator of mutable references into the sliced view
    /// of the array.
    ///
    /// Iterator element type is **&'a mut A**.
    ///
    /// **Panics** if **indexes** does not match the number of array axes.
    pub fn slice_iter_mut<'a>(&'a mut self, indexes: &[Si]) -> ElementsMut<'a, A, D> where A: Clone
    {
        let mut it = self.iter_mut();
        let offset = Dimension::do_slices(&mut it.inner.dim, &mut it.inner.strides, indexes);
        unsafe {
            it.inner.ptr = it.inner.ptr.offset(offset);
        }
        it
    }

    /// Select the subview **index** along **axis** and return an iterator
    /// of the subview.
    ///
    /// Iterator element type is **&'a mut A**.
    ///
    /// **Panics** if **axis** or **index** is out of bounds.
    pub fn sub_iter_mut<'a>(&'a mut self, axis: usize, index: Ix)
        -> ElementsMut<'a, A, D> where A: Clone
    {
        let mut it = self.iter_mut();
        dimension::do_sub(&mut it.inner.dim, &mut it.inner.ptr, &it.inner.strides, axis, index);
        it
    }

    /// Return an iterator over the diagonal elements of the array.
    pub fn diag_iter_mut<'a>(&'a mut self) -> ElementsMut<'a, A, Ix> where A: Clone
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
    pub fn raw_data_mut<'a>(&'a mut self) -> &'a mut [A] where A: Clone
    {
        self.data.make_unique().as_mut_slice()
    }


    /// Transform the array into **shape**; any other shape
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
    pub fn reshape<E: Dimension>(&self, shape: E) -> Array<A, E> where A: Clone
    {
        if shape.size() != self.dim.size() {
            panic!("Incompatible sizes in reshape, attempted from: {:?}, to: {:?}",
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

    /// Perform an elementwise assigment to **self** from **other**.
    ///
    /// If their shapes disagree, **other** is broadcast to the shape of **self**.
    ///
    /// **Panics** if broadcasting isn't possible.
    pub fn assign<E: Dimension>(&mut self, other: &Array<A, E>) where A: Clone
    {
        if self.shape() == other.shape() {
            for (x, y) in self.iter_mut().zip(other.iter()) {
                *x = y.clone();
            }
        } else {
            let other_iter = other.broadcast_iter_unwrap(self.dim());
            for (x, y) in self.iter_mut().zip(other_iter) {
                *x = y.clone();
            }
        }
    }

    /// Perform an elementwise assigment to **self** from scalar **x**.
    pub fn assign_scalar(&mut self, x: &A) where A: Clone
    {
        for elt in self.raw_data_mut().iter_mut() {
            *elt = x.clone();
        }
    }
}

/// Return a zero-dimensional array with the element **x**.
pub fn arr0<A>(x: A) -> Array<A, ()>
{
    let mut v = Vec::with_capacity(1);
    v.push(x);
    unsafe { Array::from_vec_dim((), v) }
}

/// Return a one-dimensional array with elements from **xs**.
pub fn arr1<A: Clone>(xs: &[A]) -> Array<A, Ix>
{
    Array::from_vec(xs.to_vec())
}

/// Slice or fixed-size array used for array initialization
pub unsafe trait ArrInit<T> {
    fn as_init_slice(&self) -> &[T];
    fn is_fixed_size() -> bool { false }
}

unsafe impl<T> ArrInit<T> for [T]
{
    fn as_init_slice(&self) -> &[T]
    {
        self
    }
}

macro_rules! impl_arr_init {
    (__impl $n: expr) => (
        unsafe impl<T> ArrInit<T> for [T;  $n] {
            fn as_init_slice(&self) -> &[T] { &self[] }
            fn is_fixed_size() -> bool { true }
        }
    );
    () => ();
    ($n: expr, $($m:expr,)*) => (
        impl_arr_init!(__impl $n);
        impl_arr_init!($($m,)*);
    )

}

impl_arr_init!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,);

/// Return a two-dimensional array with elements from **xs**.
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
pub fn arr2<A: Clone, V: ArrInit<A>>(xs: &[V]) -> Array<A, (Ix, Ix)>
{
    // FIXME: Simplify this when V is fix size array
    let (m, n) = (xs.len() as Ix,
                  xs.get(0).map_or(0, |snd| snd.as_init_slice().len() as Ix));
    let dim = (m, n);
    let mut result = Vec::<A>::with_capacity(dim.size());
    for snd in xs.iter() {
        let snd = snd.as_init_slice();
        assert!(<V as ArrInit<A>>::is_fixed_size() || snd.len() as Ix == n);
        result.extend(snd.iter().map(|x| x.clone()))
    }
    unsafe {
        Array::from_vec_dim(dim, result)
    }
}

/// Return a three-dimensional array with elements from **xs**.
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
pub fn arr3<A: Clone, V: ArrInit<U>, U: ArrInit<A>>(xs: &[V]) -> Array<A, (Ix, Ix, Ix)>
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
        assert!(<V as ArrInit<U>>::is_fixed_size() || snd.len() as Ix == n);
        for thr in snd.iter() {
            let thr = thr.as_init_slice();
            assert!(<U as ArrInit<A>>::is_fixed_size() || thr.len() as Ix == o);
            result.extend(thr.iter().map(|x| x.clone()))
        }
    }
    unsafe {
        Array::from_vec_dim(dim, result)
    }
}


impl<A, D> Array<A, D> where
    A: Clone + Add<Output=A>,
    D: RemoveAxis,
{
    /// Return sum along **axis**.
    ///
    /// ```
    /// use ndarray::{arr0, arr1, arr2};
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert!(
    ///     a.sum(0) == arr1(&[4., 6.]) &&
    ///     a.sum(1) == arr1(&[3., 7.]) &&
    ///
    ///     a.sum(0).sum(0) == arr0(10.)
    /// );
    /// ```
    ///
    /// **Panics** if **axis** is out of bounds.
    pub fn sum(&self, axis: usize) -> Array<A, <D as RemoveAxis>::Smaller>
    {
        let n = self.shape()[axis];
        let mut res = self.subview(axis, 0);
        for i in (1..n) {
            res.iadd(&self.subview(axis, i))
        }
        res
    }
}

impl<A, D> Array<A, D> where
    A: Copy + linalg::Field,
    D: RemoveAxis,
{
    /// Return mean along **axis**.
    ///
    /// ```
    /// use ndarray::{arr1, arr2};
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [3., 4.]]);
    /// assert!(
    ///     a.mean(0) == arr1(&[2.0, 3.0]) &&
    ///     a.mean(1) == arr1(&[1.5, 3.5])
    /// );
    /// ```
    ///
    ///
    /// **Panics** if **axis** is out of bounds.
    pub fn mean(&self, axis: usize) -> Array<A, <D as RemoveAxis>::Smaller>
    {
        let n = self.shape()[axis];
        let mut sum = self.sum(axis);
        let one = libnum::one::<A>();
        let mut cnt = one;
        for _ in (1..n) {
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
    /// Return an iterator over the elements of row **index**.
    ///
    /// **Panics** if **index** is out of bounds.
    pub fn row_iter<'a>(&'a self, index: Ix) -> Elements<'a, A, Ix>
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

    /// Return an iterator over the elements of column **index**.
    ///
    /// **Panics** if **index** is out of bounds.
    pub fn col_iter<'a>(&'a self, index: Ix) -> Elements<'a, A, Ix>
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
impl<'a, A: Copy + linalg::Ring> Array<A, (Ix, Ix)>
{
    /// Perform matrix multiplication of rectangular arrays **self** and **other**.
    ///
    /// The array sizes must agree in the way that
    /// if **self** is *M* × *N*, then **other** is *N* × *K*.
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
    pub fn mat_mul(&self, other: &Array<A, (Ix, Ix)>) -> Array<A, (Ix, Ix)>
    {
        let ((m, a), (b, n)) = (self.dim, other.dim);
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

    /// Perform the matrix multiplication of the rectangular array **self** and
    /// column vector **other**.
    ///
    /// The array sizes must agree in the way that
    /// if **self** is *M* × *N*, then **other** is *N*.
    ///
    /// Return a result array with shape *M*.
    ///
    /// **Panics** if sizes are incompatible.
    pub fn mat_mul_col(&self, other: &Array<A, Ix>) -> Array<A, Ix>
    {
        let ((m, a), n) = (self.dim, other.dim);
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


impl<A: Float + PartialOrd, D: Dimension> Array<A, D>
{
    /// Return **true** if the arrays' elementwise differences are all within
    /// the given absolute tolerance.<br>
    /// Return **false** otherwise, or if the shapes disagree.
    pub fn allclose(&self, other: &Array<A, D>, tol: A) -> bool
    {
        self.shape() == other.shape() &&
        self.iter().zip(other.iter()).all(|(x, y)| (*x - *y).abs() <= tol)
    }
}


// Array OPERATORS

macro_rules! impl_binary_op(
    ($trt:ident, $mth:ident, $imethod:ident, $imth_scalar:ident) => (
impl<A, D> Array<A, D> where A: Clone + $trt<A, Output=A>, D: Dimension
{
    /// Perform an elementwise arithmetic operation between **self** and **other**,
    /// *in place*.
    ///
    /// If their shapes disagree, **other** is broadcast to the shape of **self**.
    ///
    /// **Panics** if broadcasting isn't possible.
    pub fn $imethod <E: Dimension> (&mut self, other: &Array<A, E>)
    {
        if self.dim.ndim() == other.dim.ndim() &&
            self.shape() == other.shape() {
            for (x, y) in self.iter_mut().zip(other.iter()) {
                *x = (x.clone()). $mth (y.clone());
            }
        } else {
            let other_iter = other.broadcast_iter_unwrap(self.dim());
            for (x, y) in self.iter_mut().zip(other_iter) {
                *x = (x.clone()). $mth (y.clone());
            }
        }
    }

    /// Perform an elementwise arithmetic operation between **self** and the scalar **x**,
    /// *in place*.
    pub fn $imth_scalar (&mut self, x: &A)
    {
        for elt in self.iter_mut() {
            *elt = elt.clone(). $mth (x.clone());
        }
    }
}

impl<'a, A, D, E> $trt<Array<A, E>> for Array<A, D>
where A: Clone + $trt<A, Output=A>, D: Dimension, E: Dimension
{
    type Output = Array<A, D>;
    /// Perform an elementwise arithmetic operation between **self** and **other**,
    /// and return the result.
    ///
    /// If their shapes disagree, **other** is broadcast to the shape of **self**.
    ///
    /// **Panics** if broadcasting isn't possible.
    fn $mth (mut self, other: Array<A, E>) -> Array<A, D>
    {
        // FIXME: Can we co-broadcast arrays here? And how?
        if self.shape() == other.shape() {
            for (x, y) in self.iter_mut().zip(other.iter()) {
                *x = x.clone(). $mth (y.clone());
            }
        } else {
            let other_iter = other.broadcast_iter_unwrap(self.dim());
            for (x, y) in self.iter_mut().zip(other_iter) {
                *x = x.clone(). $mth (y.clone());
            }
        }
        self
    }
}

impl<'a, A: Clone + $trt<A, Output=A>, D: Dimension, E: Dimension>
$trt<&'a Array<A, E>> for &'a Array<A, D>
{
    type Output = Array<A, D>;
    /// Perform an elementwise arithmetic operation between **self** and **other**,
    /// and return the result.
    ///
    /// If their shapes disagree, **other** is broadcast to the shape of **self**.
    ///
    /// **Panics** if broadcasting isn't possible.
    fn $mth (self, other: &'a Array<A, E>) -> Array<A, D>
    {
        // FIXME: Can we co-broadcast arrays here? And how?
        let mut result = Vec::<A>::with_capacity(self.dim.size());
        if self.shape() == other.shape() {
            for (x, y) in self.iter().zip(other.iter()) {
                result.push((x.clone()). $mth (y.clone()));
            }
        } else {
            let other_iter = other.broadcast_iter_unwrap(self.dim());
            for (x, y) in self.iter().zip(other_iter) {
                result.push((x.clone()). $mth (y.clone()));
            }
        }
        unsafe {
            Array::from_vec_dim(self.dim.clone(), result)
        }
    }
}
    );
);

impl_binary_op!(Add, add, iadd, iadd_scalar);
impl_binary_op!(Sub, sub, isub, isub_scalar);
impl_binary_op!(Mul, mul, imul, imul_scalar);
impl_binary_op!(Div, div, idiv, idiv_scalar);
impl_binary_op!(Rem, rem, irem, irem_scalar);
impl_binary_op!(BitAnd, bitand, ibitand, ibitand_scalar);
impl_binary_op!(BitOr, bitor, ibitor, ibitor_scalar);
impl_binary_op!(BitXor, bitxor, ibitxor, ibitxor_scalar);
impl_binary_op!(Shl, shl, ishl, ishl_scalar);
impl_binary_op!(Shr, shr, ishr, ishr_scalar);

impl<A: Clone + Neg<Output=A>, D: Dimension>
Array<A, D>
{
    /// Perform an elementwise negation of **self**, *in place*.
    pub fn ineg(&mut self)
    {
        for elt in self.iter_mut() {
            *elt = elt.clone().neg()
        }
    }
}

impl<A: Clone + Neg<Output=A>, D: Dimension>
Neg for Array<A, D>
{
    type Output = Self;
    /// Perform an elementwise negation of **self** and return the result.
    fn neg(mut self) -> Array<A, D>
    {
        self.ineg();
        self
    }
}

impl<A: Clone + Not<Output=A>, D: Dimension>
Array<A, D>
{
    /// Perform an elementwise unary not of **self**, *in place*.
    pub fn inot(&mut self)
    {
        for elt in self.iter_mut() {
            *elt = elt.clone().not()
        }
    }
}

impl<A: Clone + Not<Output=A>, D: Dimension>
Not for Array<A, D>
{
    type Output = Self;
    /// Perform an elementwise unary not of **self** and return the result.
    fn not(mut self) -> Array<A, D>
    {
        self.inot();
        self
    }
}

/// An iterator over the elements of an array.
///
/// Iterator element type is **&'a A**.
pub struct Elements<'a, A, D> {
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
    /// Iterator element type is **(D, &'a A)**.
    ///
    /// **Note:** the indices run over the logical dimension of the iterator,
    /// i.e. a *.slice_iter()* will yield indices relative to the slice, not the
    /// base array.
    pub fn indexed(self) -> IndexedElements<'a, A, D>
    {
        IndexedElements {
            inner: self.inner,
        }
    }
}

/// An iterator over the elements of an array.
///
/// Iterator element type is **&'a mut A**.
pub struct ElementsMut<'a, A, D> {
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
    /// Iterator element type is **(D, &'a mut A)**.
    pub fn indexed(self) -> IndexedElementsMut<'a, A, D>
    {
        IndexedElementsMut{
            inner: self.inner,
        }
    }
}

/// An iterator over the indexes and elements of an array.
///
/// Iterator element type is **(D, &'a A)**.
pub struct IndexedElements<'a, A, D> {
    inner: Baseiter<'a, A, D>,
}

/// An iterator over the indexes and elements of an array.
///
/// Iterator element type is **(D, &'a mut A)**.
pub struct IndexedElementsMut<'a, A, D> {
    inner: Baseiter<'a, A, D>,
}
