#![feature(macro_rules)]
#![feature(phase)]
#![allow(uppercase_variables)]
#![crate_name="ndarray"]
#![crate_type="dylib"]

#[phase(plugin, link)] extern crate itertools;

// NOTE: Numpy claims it does broadcasting by
// setting up iterators with inflated shapes and several 0-stride
// dimensions.

use it = itertools;

use std::fmt;
use std::kinds;
use std::mem;
use std::num;
use std::default::Default;

pub type Ix = uint;

trait Dimension : Default + Clone + Eq {
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

/// Define a sub-dimension hierarchy
trait Shrink<T: Dimension> : Dimension {
    fn from_slice(&self, ignored: uint) -> T {
        let mut tup: T = Default::default();
        {
            let mut it = tup.shape_mut().mut_iter();
            for (i, &d) in self.shape().iter().enumerate() {
                if i == ignored {
                    continue;
                }
                *it.next().unwrap() = d;
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

/// N-dimensional array
///
/// A reference counted array with Copy-on-write mutability
pub struct Array<A, D> {
    // FIXME: Unsafecell around vec needed?
    /// Rc data when used as view, Uniquely held data when being mutated
    data: std::rc::Rc<Vec<A>>,
    /// A pointer into the buffer held by data, may point anywhere
    /// in its range.
    ptr: *mut A,
    /// The size of each dimension
    dim: D,
    /// The element count stride per dimension. To be parsed as `int`.
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

    fn make_unique<'a>(&'a mut self) {
        //println!("make_unique, needs clone={}", !std::rc::is_unique(&self.data));
        let our_off = (self.ptr as int - self.data.as_ptr() as int)
            / mem::size_of::<A>() as int;
        let rvec = self.data.make_unique();
        unsafe {
            self.ptr = rvec.as_mut_ptr().offset(our_off);
        }
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
    /// Unsafe because dimension is unchecked
    pub unsafe fn from_vec_dim(dim: D, mut v: Vec<A>) -> Array<A, D> {
        let ptr = v.as_mut_ptr();
        Array{
            data: std::rc::Rc::new(v),
            ptr: ptr,
            strides: dim.default_strides(),
            dim: dim
        }
    }

    pub fn shape(&self) -> &[uint] {
        self.dim.shape()
    }

    /// Return a sliced array.
    ///
    /// `indexes` must have one element per array dimension.
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
            index: Some(Default::default()),
            life: kinds::marker::ContravariantLifetime,
        }
    }

    pub fn iter<'a>(&'a self) -> Elements<'a, A, D>
    {
        Elements { inner: self.base_iter() }
    }

    /// Collapse dimension `axis` into length one,
    /// and select the subview of `index` along that axis.
    ///
    /// Fail if `index` is past the length of the axis
    pub fn iat_sub(&mut self, axis: uint, index: uint)
    {
        do_sub(&mut self.dim, &mut self.ptr, &self.strides, axis, index)
    }
}

impl<A, E: Dimension, D: Dimension + Shrink<E>> Array<A, D> {
    /// Select the subview `index` along `axis` and return the reduced
    /// dimension array.
    pub fn at_sub(&self, axis: uint, index: uint) -> Array<A, E>
    {
        let mut res = self.clone();
        res.iat_sub(axis, index);
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
    /// Iterate over the sliced view
    pub fn slice_iter_mut<'a>(&'a mut self, indexes: &[Slice]) -> ElementsMut<'a, A, D>
    {
        let mut it = self.iter_mut();
        let offset = do_slices(&mut it.inner.dim, &mut it.inner.strides, indexes);
        unsafe {
            it.inner.ptr = it.inner.ptr.offset(offset);
        }
        it
    }

    pub fn at_mut<'a>(&'a mut self, index: D) -> Option<&'a mut A> {
        self.make_unique();
        stride_offset_checked(&self.dim, &self.strides, &index)
            .map(|offset| unsafe {
                to_ref_mut(self.ptr.offset(offset))
            })
    }

    pub fn iter_mut<'a>(&'a mut self) -> ElementsMut<'a, A, D>
    {
        self.make_unique();
        ElementsMut { inner: self.base_iter() }
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
        // FIXME: Check if contiguous,
        // if not => copy all, else just adapt strides
        if self.strides == self.dim.default_strides() {
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

    pub fn diag_iter<'a>(&'a self) -> it::Stride<'a, A> {
        let len = self.dim.shape().iter().map(|x| *x).min().unwrap_or(0);
        let stride = self.strides.shape().iter()
                        .map(|x| *x as int)
                        .fold(0i, |s, a| s + a);
        unsafe {
            stride_new(self.ptr as *const _, len, stride as int)
        }
    }

    pub fn diag(&self) -> Array<A, uint> {
        let len = self.dim.shape().iter().map(|x| *x).min().unwrap_or(0);
        let stride = self.strides.shape().iter()
                        .map(|x| *x as int)
                        .fold(0i, |s, a| s + a);
        Array {
            data: self.data.clone(),
            ptr: self.ptr,
            dim: len,
            strides: stride as uint,
        }
    }
}


fn write_rc_array<A: fmt::Show, D: Dimension>
    (view: &Array<A, D>, f: &mut fmt::Formatter) -> fmt::Result {
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
            try!(write!(f, "{:4}", elt));
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

impl<'a, A: fmt::Show, D: fmt::Show + Dimension>
fmt::Show for Array<A, D>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        match self.dim.shape() {
            [] => {
                write!(f, "{}", self.iter().next().unwrap())
            }
            [_] => {
                try!(write!(f, "["));
                for (i, elt) in self.iter().enumerate() {
                    if i != 0 {
                        try!(write!(f, ", "));
                    }
                    try!(write!(f, "{}", elt));
                }
                write!(f, "]")
            }
            _ => {
                write_rc_array(self, f)
            }
        }
    }
}

// Array OPERATORS

impl<A: PartialEq, D: Dimension>
PartialEq for Array<A, D>
{
    fn eq(&self, other: &Array<A, D>) -> bool
    {
        assert!(self.shape() == other.shape());
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

macro_rules! impl_binary_op(
    ($trt:ident, $mth:ident, $imethod:ident) => (
impl<A: Clone + $trt<A, A>, D: Dimension> Array<A, D>
{
    pub fn $imethod (&mut self, other: &Array<A, D>)
    {
        assert!(self.shape() == other.shape());
        for (x, y) in self.iter_mut().zip(other.iter()) {
            *x = (*x). $mth (y);
        }
    }
}

impl<A: Clone + $trt<A, A>, D: Dimension>
$trt<Array<A, D>, Array<A, D>> for Array<A, D>
{
    fn $mth (&self, other: &Array<A, D>) -> Array<A, D>
    {
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
impl_binary_op!(BitAnd, bitand, ibitand)
impl_binary_op!(BitOr, bitor, ibitor)
impl_binary_op!(BitXor, bitxor, ibitxor)

impl<A: Clone + Neg<A>, D: Dimension>
Array<A, D>
{
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
    fn neg(&self) -> Array<A, D>
    {
        let mut res = self.clone();
        res.ineg();
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

/// Array iterator
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

/// Array iterator
///
/// Iterator element type is `&'a mut A`
pub struct ElementsMut<'a, A, D> {
    inner: Baseiter<'a, A, D>,
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
    for (&i, &s) in izip!(index.shape().iter(),
                          strides.shape().iter()) {
        offset += i as int * s as int;
    }
    offset
}

fn stride_offset_checked<D: Dimension>(dim: &D, strides: &D, index: &D) -> Option<int>
{
    let mut offset = 0;
    for (&d, (&i, &s)) in   dim.shape().iter().zip(
                            index.shape().iter().zip(
                            strides.shape().iter()))
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
/// Description of a range of an Array dimension.
///
/// Fields are `begin`, `end` and `stride`, where
/// negative `begin` or `end` indexes are counted from the back
/// of the dimension.
///
/// If `end` is `None`, it is taken as the last index.
pub struct Slice(pub int, pub Option<int>, pub int);

/// Full range as slice.
pub static C: Slice = Slice(0, None, 1);

#[cfg(test)]
/// Parse python slice notation into `Slice`,
/// including `a:b`, `a:b:c`, `::s`, `1:`
fn parse_slice_str(s: &str) -> Slice {
    let mut sp = s.split(':');
    let fst = sp.next();
    let snd = sp.next();
    let step = sp.next();
    assert!(sp.next().is_none());
    assert!(fst.is_some() && snd.is_some());

    let a = match fst.unwrap().trim() {
        "" => 0i,
        s => from_str::<int>(s).unwrap(),
    };
    let b = match snd.unwrap().trim() {
        "" => None,
        s => Some(from_str::<int>(s).unwrap()),
    };
    let c = match step.map(|x| x.trim()) {
        None | Some("") => 1,
        Some(s) => from_str::<int>(s).unwrap(),
    };
    Slice(a, b, c)
}


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
    for (dr, sr, &slc) in izip!(dim.shape_mut().mut_iter(),
                                strides.shape_mut().mut_iter(),
                                slices.iter())
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

#[test]
fn test_matmul_rcarray()
{
    let mut A: Array<uint, (uint, uint)> = Array::zeros((2u, 3u));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let mut B: Array<uint, (uint, uint)> = Array::zeros((3u, 4u));
    for (i, elt) in B.iter_mut().enumerate() {
        *elt = i;
    }

    let c = A.mat_mul(&B);
    println!("A = \n{}", A);
    println!("B = \n{}", B);
    println!("A x B = \n{}", c);
    unsafe {
        let result = Array::from_vec_dim((2u, 4u), vec![20u, 23, 26, 29, 56, 68, 80, 92]);
        assert_eq!(c.shape(), result.shape());
        assert!(c.iter().zip(result.iter()).all(|(a,b)| a == b));
        assert!(c == result);
    }
}

#[test]
fn test_slice()
{
    let mut A: Array<uint, (uint, uint)> = Array::zeros((3u, 4u));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let vi = A.slice([Slice(1, None, 1), Slice(0, None, 2)]);
    assert_eq!(vi.shape(), &[2u, 2u]);
    let vi = A.slice([C, C]);
    assert_eq!(vi.shape(), A.shape());
    assert!(vi.iter().zip(A.iter()).all(|(a, b)| a == b));
}

#[test]
fn test_index()
{
    let mut A: Array<uint, (uint, uint)> = Array::zeros((2u, 3u));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    for ((i, j), x) in iproduct!(range(0,2u), range(0,3u)).zip(A.iter()) {
        assert_eq!(*x, A[(i, j)]);
    }

    let vi = A.slice([Slice(1, None, 1), Slice(0, None, 2)]);
    let mut it = vi.iter();
    for (i, j) in iproduct!(range(0, 1u), range(0, 2u)) {
        let x = it.next().unwrap();
        assert_eq!(*x, vi[(i, j)]);
    }
    assert!(it.next().is_none());
}

#[test]
fn test_add()
{
    let mut A: Array<uint, (uint, uint)> = Array::zeros((2u, 2u));
    for (i, elt) in A.iter_mut().enumerate() {
        *elt = i;
    }

    let B = A.clone();
    A.iadd(&B);
    assert_eq!(A[(0,0)], 0u);
    assert_eq!(A[(0,1)], 2u);
    assert_eq!(A[(1,0)], 4u);
    assert_eq!(A[(1,1)], 6u);
}

#[test]
fn test_parse()
{
    let slice_strings = ["1:2:3", "::", "1:", "::-1", "::2"];
    for s in slice_strings.iter() {
        println!("Parse {} \t=> {}", *s, parse_slice_str(*s));
    }
}

#[test]
fn test_multidim()
{
    let mut mat = Array::zeros(2u*3*4*5*6).reshape((2u,3u,4u,5u,6u));
    mat[(0,0,0,0,0)] = 22u8;
    {
        for (i, elt) in mat.iter_mut().enumerate() {
            *elt = i as u8;
        }
    }
    println!("shape={}, strides={}", mat.shape(), mat.strides);
    assert_eq!(mat.shape(), &[2u,3,4,5,6]);
}


/*
array([[[ 7,  6],
        [ 5,  4],
        [ 3,  2],
        [ 1,  0]],

       [[15, 14],
        [13, 12],
        [11, 10],
        [ 9,  8]]])
*/
#[test]
fn test_negative_stride_rcarray()
{
    let mut mat = Array::zeros((2u, 4u, 2u));
    mat[(0, 0, 0)] = 1.0f32;
    for (i, elt) in mat.iter_mut().enumerate() {
        *elt = i as f32;
    }

    {
        let vi = mat.slice([C, Slice(0, None, -1), Slice(0, None, -1)]);
        assert_eq!(vi.shape(), &[2,4,2]);
        // Test against sequential iterator
        let seq = [7f32,6., 5.,4.,3.,2.,1.,0.,15.,14.,13., 12.,11.,  10.,   9.,   8.];
        for (a, b) in vi.clone().iter().zip(seq.iter()) {
            assert_eq!(*a, *b);
        }
    }
    {
        let vi = mat.slice([C, Slice(0, None, -5), C]);
        let seq = [6_f32, 7., 14., 15.];
        for (a, b) in vi.iter().zip(seq.iter()) {
            assert_eq!(*a, *b);
        }
    }
}

#[test]
fn test_cow()
{
    let mut mat = Array::<int, _>::zeros((2u,2u));
    mat[(0, 0)] = 1;
    let n = mat.clone();
    mat[(0, 1)] = 2;
    mat[(1, 0)] = 3;
    mat[(1, 1)] = 4;
    assert_eq!(mat[(0,0)], 1);
    assert_eq!(mat[(0,1)], 2);
    assert_eq!(n[(0,0)], 1);
    assert_eq!(n[(0,1)], 0);
    let mut rev = mat.reshape(4u).slice([Slice(0, None, -1)]);
    assert_eq!(rev[0], 4);
    assert_eq!(rev[1], 3);
    assert_eq!(rev[2], 2);
    assert_eq!(rev[3], 1);
    let before = rev.clone();
    // mutation
    rev[0] = 5;
    assert_eq!(rev[0], 5);
    assert_eq!(rev[1], 3);
    assert_eq!(rev[2], 2);
    assert_eq!(rev[3], 1);
    assert_eq!(before[0], 4);
    assert_eq!(before[1], 3);
    assert_eq!(before[2], 2);
    assert_eq!(before[3], 1);
}

#[test]
fn test_sub()
{
    let mat = Array::from_iter(range(0.0f32, 16.0)).reshape((2u, 4u, 2u));
    let s1 = mat.at_sub(0,0);
    let s2 = mat.at_sub(0,1);
    assert_eq!(s1.shape(), &[4, 2]);
    assert_eq!(s2.shape(), &[4, 2]);
    let n = Array::from_iter(range(8.0f32, 16.0)).reshape((4u,2u));
    assert_eq!(n, s2);
    let m = Array::from_vec(vec![2f32, 3., 10., 11.]).reshape((2u, 2u));
    assert_eq!(m, mat.at_sub(1, 1));
}
