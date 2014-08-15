#![feature(macro_rules)]
#![feature(default_type_params)] /* Hash<S> */
#![crate_name="ndarray"]
#![crate_type="dylib"]

//! The **ndarray** crate provides the **Array** type, an n-dimensional
//! numerical container similar to numpy's ndarray.
//!

extern crate itertools;

use itertools::ItertoolsClonable;
use it = itertools;

use std::fmt;
use std::hash;
use std::kinds;
use std::mem;
use std::num;
use std::default::Default;

/// Array index type
pub type Ix = uint;

/// Trait for the shape and index types of arrays.
pub trait Dimension : Clone + Eq {
    fn ndim(&self) -> uint;
    fn shape<'a>(&'a self) -> &'a [Ix] {
        unsafe {
            std::mem::transmute(std::raw::Slice {
                data: self as *const _ as *const Ix,
                len: self.ndim(),
            })
        }
    }

    fn shape_mut<'a>(&'a mut self) -> &'a mut [Ix] {
        unsafe {
            std::mem::transmute(std::raw::Slice {
                data: self as *mut _ as *const Ix,
                len: self.ndim(),
            })
        }
    }

    fn size(&self) -> uint {
        self.shape().iter().fold(1u, |s, &a| s * a as uint)
    }

    fn default_strides(&self) -> Self {
        // Compute default array strides
        // Shape (a, b, c) => Give strides (b * c, c, 1)
        let mut strides = self.clone();
        {
            let mut it = strides.shape_mut().mut_iter().rev();
            // Set first element to 1
            for rs in it {
                *rs = 1;
                break;
            }
            let mut cum_prod = 1;
            for (rs, dim) in it.zip(self.shape().iter().rev()) {
                cum_prod *= *dim;
                *rs = cum_prod;
            }
        }
        strides
    }

    fn first_index(&self) -> Self
    {
        let mut index = self.clone();
        for rr in index.shape_mut().mut_iter() {
            *rr = 0;
        }
        index
    }

    /// Iteration -- Use self as size, and return next index after `index`
    /// or None if there are no more.
    fn next_for(&self, index: Self) -> Option<Self> {
        let mut index = index;
        let mut done = false;
        for (&dim, ix) in self.shape().iter().rev()
                            .zip(index.shape_mut().mut_iter().rev())
        {
            *ix += 1;
            if *ix == dim {
                *ix = 0;
            } else {
                done = true;
                break;
            }
        }
        if done {
            Some(index)
        } else { None }
    }
}

impl Dimension for () {
    // empty product is 1 -> size is 1
    fn ndim(&self) -> uint { 0 }
    fn shape(&self) -> &[Ix] { &[] }
    fn shape_mut(&mut self) -> &mut [Ix] { &mut [] }
}

impl Dimension for Ix {
    fn ndim(&self) -> uint { 1 }
    fn next_for(&self, mut index: Ix) -> Option<Ix> {
        index += 1;
        if index < *self {
            Some(index)
        } else { None }
    }
}

impl Dimension for (Ix, Ix) {
    fn ndim(&self) -> uint { 2 }
    fn next_for(&self, index: (Ix, Ix)) -> Option<(Ix, Ix)> {
        let (mut i, mut j) = index;
        let (imax, jmax) = *self;
        j += 1;
        if j == jmax {
            j = 0;
            i += 1;
            if i == imax {
                return None;
            }
        }
        Some((i, j))
    }
}

impl Dimension for (Ix, Ix, Ix) {
    fn ndim(&self) -> uint { 3 }
    fn next_for(&self, index: (Ix, Ix, Ix)) -> Option<(Ix, Ix, Ix)> {
        let (mut i, mut j, mut k) = index;
        let (imax, jmax, kmax) = *self;
        k += 1;
        if k == kmax {
            k = 0;
            j += 1;
            if j == jmax {
                j = 0;
                i += 1;
                if i == imax {
                    return None;
                }
            }
        }
        Some((i, j, k))
    }
}

impl Dimension for (Ix, Ix, Ix, Ix) { fn ndim(&self) -> uint { 4 } }
impl Dimension for (Ix, Ix, Ix, Ix, Ix) { fn ndim(&self) -> uint { 5 } }
impl Dimension for (Ix, Ix, Ix, Ix, Ix, Ix) { fn ndim(&self) -> uint { 6 } }
impl Dimension for (Ix, Ix, Ix, Ix, Ix, Ix, Ix) { fn ndim(&self) -> uint { 7 } }
impl Dimension for (Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix) { fn ndim(&self) -> uint { 8 } }
impl Dimension for (Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix) { fn ndim(&self) -> uint { 9 } }
impl Dimension for (Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix) { fn ndim(&self) -> uint { 10 } }
impl Dimension for (Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix) { fn ndim(&self) -> uint { 11 } }
impl Dimension for (Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix) { fn ndim(&self) -> uint { 12 } }

// Vec<Ix> is a "dynamic" index, pretty hard to use when indexing,
// and memory wasteful, but it allows an arbitrary number of dimensions.
//
// NOTE: No Shrink impl for Vec<Ix> yet.
impl Dimension for Vec<Ix>
{
    fn ndim(&self) -> uint { self.len() }
    fn shape(&self) -> &[Ix] { self.as_slice() }
    fn shape_mut(&mut self) -> &mut [Ix] { self.as_mut_slice() }
}

/// Helper trait to define a smaller-than relation for array shapes.
trait Shrink<T: Dimension + Default> : Dimension {
    fn from_slice(&self, ignored: uint) -> T {
        let mut tup: T = Default::default();
        {
            let mut it = tup.shape_mut().mut_iter();
            for (i, &d) in self.shape().iter().enumerate() {
                if i == ignored {
                    continue;
                }
                for rr in it {
                    *rr = d;
                    break
                }
            }
        }
        tup
    }
}

impl Shrink<()> for Ix { }
impl Shrink<Ix> for (Ix, Ix) { }
impl Shrink<(Ix, Ix)> for (Ix, Ix, Ix) { }
impl Shrink<(Ix, Ix, Ix)> for (Ix, Ix, Ix, Ix) { }
impl Shrink<(Ix, Ix, Ix, Ix)> for (Ix, Ix, Ix, Ix, Ix) { }
impl Shrink<(Ix, Ix, Ix, Ix, Ix)> for (Ix, Ix, Ix, Ix, Ix, Ix) { }
impl Shrink<(Ix, Ix, Ix, Ix, Ix, Ix)> for (Ix, Ix, Ix, Ix, Ix, Ix, Ix) { }
impl Shrink<(Ix, Ix, Ix, Ix, Ix, Ix, Ix)> for (Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix) { }
impl Shrink<(Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix)> for (Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix) { }
impl Shrink<(Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix)> for (Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix) { }
impl Shrink<(Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix)> for (Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix) { }
impl Shrink<(Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix)> for (Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix) { }

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

    fn make_unique<'a>(&'a mut self) -> &'a mut Vec<A>
    {
        let our_off = (self.ptr as int - self.data.as_ptr() as int)
            / mem::size_of::<A>() as int;
        let rvec = self.data.make_unique();
        unsafe {
            self.ptr = rvec.as_mut_ptr().offset(our_off);
        }
        rvec
    }
}

impl<A> Array<A, uint>
{
    /// Create a one-dimensional array from a vector (no allocation needed)
    pub fn from_vec(v: Vec<A>) -> Array<A, uint> {
        unsafe {
            Array::from_vec_dim(v.len(), v)
        }
    }

    /// Create a one-dimensional array from an iterator
    pub fn from_iter<I: Iterator<A>>(mut it: I) -> Array<A, uint> {
        Array::from_vec(it.collect())
    }

}

impl<A: Clone> Array<A, uint>
{
    /// Create a one-dimensional array from a slice
    pub fn from_slice(s: &[A]) -> Array<A, uint>
    {
        Array::from_vec(s.to_vec())
    }
}

impl<A: Clone> Array<A, (uint, uint)>
{
    /// Create a two-dimensional array from a slice
    ///
    /// Fail if the slices are not all of the same length.
    ///
    /// ```
    /// use ndarray::Array;
    /// let a = Array::from_slices([[1, 2, 3],
    ///                             [4, 5, 6i]]);
    /// assert!(a.dim() == (2, 3));
    /// ```
    pub fn from_slices(s: &[&[A]]) -> Array<A, (uint, uint)>
    {
        unsafe {
            match s.get(0).map(|t| t.len()) {
                None => Array::from_vec_dim((0u, 0u), Vec::new()),
                Some(n) => {
                    assert!(s.iter().all(|l| l.len() == n));
                    let m = s.len();
                    let v = s.iter().flat_map(|l| l.iter()).clones().collect::<Vec<A>>();
                    Array::from_vec_dim((m, n), v)
                }
            }
        }
    }
}

/// Collapse axis `axis` and shift so that only subarray `index` is
/// available.
///
/// Fails if `index` is larger than the size of the axis
fn do_sub<A, D: Dimension, P: Copy + RawPtr<A>>(dims: &mut D, ptr: &mut P, strides: &D,
                           axis: uint, index: uint)
{
    let dim = dims.shape()[axis];
    let stride = strides.shape()[axis] as int;
    assert!(index < dim);
    dims.shape_mut()[axis] = 1;
    let off = stride * index as int;
    unsafe {
        *ptr = ptr.offset(off);
    }
}

impl<A, D: Dimension> Array<A, D>
{
    /// Unsafe because dimension is unchecked.
    pub unsafe fn from_vec_dim(dim: D, mut v: Vec<A>) -> Array<A, D> {
        let ptr = v.as_mut_ptr();
        Array{
            data: std::rc::Rc::new(v),
            ptr: ptr,
            strides: dim.default_strides(),
            dim: dim
        }
    }

    pub fn dim(&self) -> D {
        self.dim.clone()
    }

    pub fn shape(&self) -> &[uint] {
        self.dim.shape()
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
    /// `indexes` must have one element per array axis.
    pub fn slice(&self, indexes: &[Slice]) -> Array<A, D>
    {
        let mut arr = self.clone();
        arr.islice(indexes);
        arr
    }

    /// Like `slice`, except this array's view is mutated in place
    pub fn islice(&mut self, indexes: &[Slice])
    {
        let offset = do_slices(&mut self.dim, &mut self.strides, indexes);
        unsafe {
            self.ptr = self.ptr.offset(offset);
        }
    }

    /// Iterate over the sliced view
    pub fn slice_iter<'a>(&'a self, indexes: &[Slice]) -> Elements<'a, A, D>
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
        stride_offset_checked(&self.dim, &self.strides, &index)
            .map(|offset| unsafe {
                to_ref(self.ptr.offset(offset) as *const _)
            })
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
    /// Iterator element type is `&'a A`
    pub fn iter<'a>(&'a self) -> Elements<'a, A, D>
    {
        Elements { inner: self.base_iter() }
    }

    /// Collapse dimension `axis` into length one,
    /// and select the subview of `index` along that axis.
    ///
    /// Fail if `index` is past the length of the axis
    pub fn isubview(&mut self, axis: uint, index: uint)
    {
        do_sub(&mut self.dim, &mut self.ptr, &self.strides, axis, index)
    }

    /// Act like a larger size and/or dimension Array by *broadcasting*
    /// into a larger shape, if possible.
    ///
    /// Return `None` if not compatible.
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
            if to.shape().len() < from.shape().len() {
                return None
            }

            {
                let mut new_stride_iter = new_stride.shape_mut().mut_iter().rev();
                for ((er, es), dr) in from.shape().iter().rev()
                                        .zip(stride.shape().iter().rev())
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
    /// Fail if the axes are out of bounds.
    pub fn swap_axes(&mut self, ax: uint, bx: uint)
    {
        self.dim.shape_mut().swap(ax, bx);
        self.strides.shape_mut().swap(ax, bx);
    }
}

impl<A, E: Dimension + Default, D: Dimension + Shrink<E>> Array<A, D>
{
    /// Select the subview `index` along `axis` and return the reduced
    /// dimension array.
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
    /// Return a mutable reference to the element at `index`, or return `None`
    /// if the index is out of bounds.
    pub fn at_mut<'a>(&'a mut self, index: D) -> Option<&'a mut A> {
        self.make_unique();
        stride_offset_checked(&self.dim, &self.strides, &index)
            .map(|offset| unsafe {
                to_ref_mut(self.ptr.offset(offset))
            })
    }

    /// Return an iterator of mutable references to the elements of the Array
    ///
    /// Iterator element type is `&'a mut A`
    pub fn iter_mut<'a>(&'a mut self) -> ElementsMut<'a, A, D>
    {
        self.make_unique();
        ElementsMut { inner: self.base_iter(), nocopy: kinds::marker::NoCopy }
    }

    /// Return an iterator of mutable references into the sliced view
    /// of the array.
    ///
    /// Iterator element type is `&'a mut A`
    pub fn slice_iter_mut<'a>(&'a mut self, indexes: &[Slice]) -> ElementsMut<'a, A, D>
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
    /// Iterator element type is `&'a mut A`
    pub fn sub_iter_mut<'a>(&'a mut self, axis: uint, index: uint)
        -> ElementsMut<'a, A, D>
    {
        let mut it = self.iter_mut();
        do_sub(&mut it.inner.dim, &mut it.inner.ptr, &it.inner.strides, axis, index);
        it
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
        self.make_unique().as_mut_slice()
    }


    /// Transform the array into `shape`, must correspond
    /// to the same number of elements.
    ///
    /// fail on incompatible size.
    pub fn reshape<E: Dimension>(&self, shape: E) -> Array<A, E> {
        if shape.size() != self.dim.size() {
            fail!("Incompatible sizes in reshape, attempted from: {}, to: {}",
                  self.dim.shape(), shape.shape())
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
    /// Fails if broadcasting isn't possible.
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
    let begin;
    let end;
    if len != 0 {
        begin = ptr as *const _;
        end = begin.offset((len - 1) as int * stride);
    } else {
        begin = std::ptr::null();
        end = std::ptr::null();
    }
    it::Stride::from_ptrs(begin, end, stride)
}


impl<A> Array<A, (Ix, Ix)>
{
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

impl<A, D: Dimension> Array<A, D>
{
    pub fn iter1d<'b>(&'b self, axis: uint, from: &D) -> it::Stride<'b, A> {
        let dim = self.dim.shape()[axis];
        let stride = self.strides.shape()[axis];
        let off = stride_offset_checked(&self.dim, &self.strides, from).unwrap();
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
        let len = self.dim.shape().iter().clones().min().unwrap_or(1);
        let stride = self.strides.shape().iter()
                        .map(|x| *x as int)
                        .fold(0i, |s, a| s + a);
        return (len, stride)
    }

    pub fn diag_iter<'a>(&'a self) -> it::Stride<'a, A> {
        let (len, stride) = self.diag_params();
        unsafe {
            stride_new(self.ptr as *const _, len, stride)
        }
    }

    pub fn diag(&self) -> Array<A, uint> {
        let (len, stride) = self.diag_params();
        Array {
            data: self.data.clone(),
            ptr: self.ptr,
            dim: len,
            strides: stride as uint,
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
    Array::from_slice(xs)
}

/// Return a two-dimensional array with elements from `xs`.
///
/// Fail if the slices are not all of the same length.
pub fn arr2<A: Clone>(xs: &[&[A]]) -> Array<A, (Ix, Ix)>
{
    Array::from_slices(xs)
}

impl<A: Clone + Num,
     E: Dimension + Default, D: Dimension + Shrink<E>>
    Array<A, D>
{
    /// Return sum along `axis`
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
    pub fn sum(&self, axis: uint) -> Array<A, E>
    {
        let n = self.shape()[axis];
        let mut res = self.subview(axis, 0);
        for i in range(1, n) {
            res.iadd(&self.subview(axis, i))
        }
        res
    }

    /// Return mean along `axis`
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

fn format_array<A, D: Dimension>(view: &Array<A, D>, f: &mut fmt::Formatter,
                                 format: |&mut fmt::Formatter, &A| -> fmt::Result)
                                -> fmt::Result
{
    if f.width.is_none() {
        f.width = Some(4)
    }
    match view.dim.shape() {
        [] => {
            return format(f, view.iter().next().unwrap());
        }
        [_] => {
            try!(write!(f, "["));
            for (i, elt) in view.iter().enumerate() {
                if i != 0 {
                    try!(write!(f, ", "));
                }
                try!(format(f, elt));
            }
            return write!(f, "]");
        }
        _ => /* fallthrough */ {}
    }
    let mut slices = Vec::from_elem(view.dim.shape().len(), C);
    assert!(slices.len() >= 2);
    let n_loops = slices.len() - 2;
    let row_width = view.dim.shape()[slices.len() - 1];
    let mut fixed = Vec::from_elem(n_loops, 0u);
    let mut first = true;
    loop {
        /* Use fixed indices to make a slice*/
        for (fidx, slc) in fixed.iter().zip(slices.mut_iter()) {
            *slc = Slice(*fidx as int, Some(*fidx as int + 1), 1);
        }
        if !first {
            try!(write!(f, "\n"));
        }
        /* Print out this view */
        for (i, elt) in view.slice_iter(slices.as_slice()).enumerate() {
            if i % row_width != 0 {
                try!(write!(f, ", "));
            } else if i != 0 {
                try!(write!(f, "\n ["));
            } else {
                try!(write!(f, "[["));
            }
            //try!(write!(f, "{:4}", elt));
            try!(format(f, elt));
            if i != 0 && (i+1) % row_width == 0 {
                try!(write!(f, "]"));
            }
        }
        first = false;
        try!(write!(f, "]"));
        let mut done = true;
        for (fidx, &dim) in fixed.mut_iter().zip(view.dim.shape().iter()) {
            *fidx += 1;
            if *fidx == dim {
                *fidx = 0;
                continue;
            } else {
                done = false;
                break;
            }
        }
        if done {
            break
        }
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
    /// Fail if shapes are not equal.
    fn eq(&self, other: &Array<A, D>) -> bool
    {
        assert!(self.shape() == other.shape());
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<A: Eq, D: Dimension>
Eq for Array<A, D> {}

macro_rules! impl_binary_op(
    ($trt:ident, $mth:ident, $imethod:ident) => (
impl<A: Clone + $trt<A, A>, D: Dimension, E: Dimension>
Array<A, D>
{
    /// Perform an elementwise arithmetic operation between `self` and `other`,
    /// *in place*.
    ///
    /// If their shapes disagree, `other` is broadcast to the shape of `self`.
    /// Fails if broadcasting isn't possible.
    pub fn $imethod (&mut self, other: &Array<A, E>)
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
}

impl<A: Clone + $trt<A, A>, D: Dimension, E: Dimension>
$trt<Array<A, E>, Array<A, D>> for Array<A, D>
{
    /// Perform an elementwise arithmetic operation between `self` and `other`,
    /// and return the result.
    ///
    /// If their shapes disagree, `other` is broadcast to the shape of `self`.
    /// Fails if broadcasting isn't possible.
    fn $mth (&self, other: &Array<A, E>) -> Array<A, D>
    {
        // FIXME: Can we co-broadcast arrays here? And how?
        let mut res = self.clone();
        res.$imethod (other);
        res
    }
}
    );
)

impl_binary_op!(Add, add, iadd)
impl_binary_op!(Sub, sub, isub)
impl_binary_op!(Mul, mul, imul)
impl_binary_op!(Div, div, idiv)
impl_binary_op!(Rem, rem, irem)
impl_binary_op!(BitAnd, bitand, ibitand)
impl_binary_op!(BitOr, bitor, ibitor)
impl_binary_op!(BitXor, bitxor, ibitxor)
impl_binary_op!(Shl, shl, ishl)
impl_binary_op!(Shr, shr, ishr)

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
/// Iterator element type is `&'a A`
struct Baseiter<'a, A, D> {
    ptr: *mut A,
    dim: D,
    strides: D,
    index: Option<D>,
    life: kinds::marker::ContravariantLifetime<'a>,
}

impl<'a, A, D: Dimension> Baseiter<'a, A, D>
{
    fn next(&mut self) -> Option<*mut A>
    {
        let index = match self.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        let offset = stride_offset(&self.strides, &index);
        self.index = self.dim.next_for(index);
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

#[deriving(Clone)]
/// An iterator over the elements of an array.
///
/// Iterator element type is `&'a A`
pub struct Elements<'a, A, D> {
    inner: Baseiter<'a, A, D>,
}

impl<'a, A, D: Dimension> Iterator<&'a A> for Elements<'a, A, D>
{
    fn next(&mut self) -> Option<&'a A>
    {
        unsafe {
            self.inner.next().map(|p| to_ref(p as *const _))
        }
    }
}

/// An iterator over the elements of an array.
///
/// Iterator element type is `&'a mut A`
pub struct ElementsMut<'a, A, D> {
    inner: Baseiter<'a, A, D>,
    nocopy: kinds::marker::NoCopy,
}

impl<'a, A, D: Dimension> Iterator<&'a mut A> for ElementsMut<'a, A, D>
{
    fn next(&mut self) -> Option<&'a mut A>
    {
        unsafe {
            self.inner.next().map(|p| to_ref_mut(p))
        }
    }
}

fn stride_offset<D: Dimension>(strides: &D, index: &D) -> int
{
    let mut offset = 0;
    for (&i, &s) in index.shape().iter()
                        .zip(strides.shape().iter()) {
        offset += i as int * s as int;
    }
    offset
}

fn stride_offset_checked<D: Dimension>(dim: &D, strides: &D, index: &D) -> Option<int>
{
    let mut offset = 0;
    for ((&d, &i), &s) in dim.shape().iter()
                            .zip(index.shape().iter())
                            .zip(strides.shape().iter())
    {
        if i >= d {
            return None;
        }
        offset += i as int * s as int;
    }
    Some(offset)
}


// [a:b:s] syntax for example [:3], [::-1]
// [0,:] -- first row of matrix
// [:,0] -- first column of matrix

#[deriving(Clone, PartialEq, Eq, Hash, Show)]
/// Description of a range of an array axis.
///
/// Fields are `begin`, `end` and `stride`, where
/// negative `begin` or `end` indexes are counted from the back
/// of the axis.
///
/// If `end` is `None`, the slice extends to the end of the axis.
///
/// ## Examples
///
/// `Slice(0, None, 1)` is the full range of an axis.
/// Python equivalent is `[:]`.
///
/// `Slice(a, Some(b), 2)` is every second element from `a` until `b`.
/// Python equivalent is `[a:b:2]`.
///
/// `Slice(a, None, -1)` is every element, in reverse order, from `a`
/// until the end. Python equivalent is `[a::-1]`
pub struct Slice(pub int, pub Option<int>, pub int);

/// Full range as slice.
pub static C: Slice = Slice(0, None, 1);

fn abs_index(len: int, index: int) -> uint {
    if index < 0 {
        (len + index) as uint
    } else { index as uint }
}

/// Modify dimension, strides and return data pointer offset
fn do_slices<D: Dimension>(dim: &mut D, strides: &mut D, slices: &[Slice]) -> int
{
    let mut offset = 0;
    assert!(slices.len() == dim.shape().len());
    for ((dr, sr), &slc) in dim.shape_mut().mut_iter()
                            .zip(strides.shape_mut().mut_iter())
                            .zip(slices.iter())
    {
        let m = *dr;
        let mi = m as int;
        let Slice(b1, opt_e1, s1) = slc;
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


// Matrix multiplication only defined for Primitive to
// avoid trouble with failing + and *
impl<'a, A: Primitive> Array<A, (Ix, Ix)>
{
    /// Matrix multiplication of arrays `self` and `other`
    ///
    /// The array sizes must agree in the way that
    /// `self` is M x N  and `other` is N x K, the result then being
    /// size M x K
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
            let row = self.row_iter(i);
            let col = other.col_iter(j);
            let dot = row.zip(col).fold(num::zero(), |s: A, (x, y)| {
                    s + *x * *y
                });
            unsafe {
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
}

impl<A: Clone> FromIterator<A> for Array<A, uint>
{
    fn from_iter<I: Iterator<A>>(it: I) -> Array<A, uint>
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
