use std::mem;
use std::num::SignedInt;
use std::raw;

use super::{Ix, Ixs};

/// Calculate offset from `Ix` stride converting sign properly
#[inline]
pub fn stride_offset(n: Ix, stride: Ix) -> int
{
    (n as int) * ((stride as Ixs) as int)
}

/*
#[inline]
pub fn stride_as_int(stride: Ix) -> int
{
    (stride as Ixs) as int
}
*/

/// Trait for the shape and index types of arrays.
pub trait Dimension : Clone + Eq {
    fn ndim(&self) -> uint;
    fn slice<'a>(&'a self) -> &'a [Ix] {
        unsafe {
            mem::transmute(raw::Slice {
                data: self as *const _ as *const Ix,
                len: self.ndim(),
            })
        }
    }

    fn slice_mut<'a>(&'a mut self) -> &'a mut [Ix] {
        unsafe {
            mem::transmute(raw::Slice {
                data: self as *mut _ as *const Ix,
                len: self.ndim(),
            })
        }
    }

    fn size(&self) -> uint {
        self.slice().iter().fold(1u, |s, &a| s * a as uint)
    }

    fn default_strides(&self) -> Self {
        // Compute default array strides
        // Shape (a, b, c) => Give strides (b * c, c, 1)
        let mut strides = self.clone();
        {
            let mut it = strides.slice_mut().iter_mut().rev();
            // Set first element to 1
            for rs in it {
                *rs = 1;
                break;
            }
            let mut cum_prod = 1;
            for (rs, dim) in it.zip(self.slice().iter().rev()) {
                cum_prod *= *dim;
                *rs = cum_prod;
            }
        }
        strides
    }

    #[inline]
    fn first_index(&self) -> Option<Self>
    {
        for ax in self.slice().iter() {
            if *ax == 0 {
                return None
            }
        }
        let mut index = self.clone();
        for rr in index.slice_mut().iter_mut() {
            *rr = 0;
        }
        Some(index)
    }

    /// Iteration -- Use self as size, and return next index after `index`
    /// or None if there are no more.
    // FIXME: use &Self for index or even &mut?
    #[inline]
    fn next_for(&self, index: Self) -> Option<Self> {
        let mut index = index;
        let mut done = false;
        for (&dim, ix) in self.slice().iter().rev()
                            .zip(index.slice_mut().iter_mut().rev())
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

    /// Return stride offset for index.
    fn stride_offset(index: &Self, strides: &Self) -> int
    {
        let mut offset = 0;
        for (&i, &s) in index.slice().iter()
                            .zip(strides.slice().iter()) {
            offset += stride_offset(i, s);
        }
        offset
    }

    /// Return stride offset for this dimension and index.
    fn stride_offset_checked(&self, strides: &Self, index: &Self) -> Option<int>
    {
        let mut offset = 0;
        for ((&d, &i), &s) in self.slice().iter()
                                .zip(index.slice().iter())
                                .zip(strides.slice().iter())
        {
            if i >= d {
                return None;
            }
            offset += stride_offset(i, s);
        }
        Some(offset)
    }

    /// Modify dimension, strides and return data pointer offset
    ///
    /// **Panics** if `slices` does not correspond to the number of axes,
    /// if any stride is 0, or if any index is out of bounds.
    fn do_slices(dim: &mut Self, strides: &mut Self, slices: &[Si]) -> int
    {
        let mut offset = 0;
        assert!(slices.len() == dim.slice().len());
        for ((dr, sr), &slc) in dim.slice_mut().iter_mut()
                                .zip(strides.slice_mut().iter_mut())
                                .zip(slices.iter())
        {
            let m = *dr;
            let mi = m as Ixs;
            let Si(b1, opt_e1, s1) = slc;
            let e1 = opt_e1.unwrap_or(mi);

            let b1 = abs_index(mi, b1);
            let mut e1 = abs_index(mi, e1);
            if e1 < b1 { e1 = b1; }

            assert!(b1 <= m);
            assert!(e1 <= m);

            let m = e1 - b1;
            // stride
            let s = (*sr) as Ixs;

            // Data pointer offset
            offset += stride_offset(b1, *sr);
            // Adjust for strides
            assert!(s1 != 0);
            // How to implement negative strides:
            //
            // Increase start pointer by
            // old stride * (old dim - 1)
            // to put the pointer completely in the other end
            if s1 < 0 {
                offset += stride_offset(m - 1, *sr);
            }

            let s_prim = s * s1;

            let d = m / s1.abs() as Ix;
            let r = m % s1.abs() as Ix;
            let m_prim = d + if r > 0 { 1 } else { 0 };

            // Update dimension and stride coordinate
            *dr = m_prim;
            *sr = s_prim as Ix;
        }
        offset
    }
}

fn abs_index(len: Ixs, index: Ixs) -> Ix {
    if index < 0 {
        (len + index) as Ix
    } else { index as Ix }
}

/// Collapse axis `axis` and shift so that only subarray `index` is
/// available.
///
/// **Panics** if `index` is larger than the size of the axis
// FIXME: Move to Dimension trait
pub fn do_sub<A, D: Dimension, P: Copy + RawPtr<A>>(dims: &mut D, ptr: &mut P, strides: &D,
                           axis: uint, index: Ix)
{
    let dim = dims.slice()[axis];
    let stride = strides.slice()[axis];
    assert!(index < dim);
    dims.slice_mut()[axis] = 1;
    let off = stride_offset(index, stride);
    unsafe {
        *ptr = ptr.offset(off);
    }
}


impl Dimension for () {
    // empty product is 1 -> size is 1
    #[inline]
    fn ndim(&self) -> uint { 0 }
    fn slice(&self) -> &[Ix] { &[] }
    fn slice_mut(&mut self) -> &mut [Ix] { &mut [] }
}

impl Dimension for Ix {
    #[inline]
    fn ndim(&self) -> uint { 1 }
    #[inline]
    fn size(&self) -> uint { *self as uint }
    #[inline]
    fn first_index(&self) -> Option<Ix> {
        if *self != 0 {
            Some(0)
        } else { None }
    }
    #[inline]
    fn next_for(&self, mut index: Ix) -> Option<Ix> {
        index += 1;
        if index < *self {
            Some(index)
        } else { None }
    }

    /// Self is an index, return the stride offset
    #[inline]
    fn stride_offset(index: &Ix, stride: &Ix) -> int
    {
        stride_offset(*index, *stride)
    }

    /// Return stride offset for this dimension and index.
    fn stride_offset_checked(&self, stride: &Ix, index: &Ix) -> Option<int>
    {
        if *index < *self {
            Some(stride_offset(*index, *stride))
        } else {
            None
        }
    }
}

impl Dimension for (Ix, Ix) {
    #[inline]
    fn ndim(&self) -> uint { 2 }
    #[inline]
    fn size(&self) -> uint { let (m, n) = *self; m as uint * n as uint }
    #[inline]
    fn first_index(&self) -> Option<(Ix, Ix)> {
        let (m, n) = *self;
        if m != 0 && n != 0 {
            Some((0, 0))
        } else { None }
    }
    #[inline]
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

    /// Self is an index, return the stride offset
    #[inline]
    fn stride_offset(index: &(Ix, Ix), strides: &(Ix, Ix)) -> int
    {
        let (i, j) = *index;
        let (s, t) = *strides;
        stride_offset(i, s) + stride_offset(j, t)
    }

    /// Return stride offset for this dimension and index.
    fn stride_offset_checked(&self, strides: &(Ix, Ix), index: &(Ix, Ix)) -> Option<int>
    {
        let (m, n) = *self;
        let (i, j) = *index;
        let (s, t) = *strides;
        if i < m && j < n {
            Some(stride_offset(i, s) + stride_offset(j, t))
        } else {
            None
        }
    }
}

impl Dimension for (Ix, Ix, Ix) {
    #[inline]
    fn ndim(&self) -> uint { 3 }
    #[inline]
    fn size(&self) -> uint { let (m, n, o) = *self; m as uint * n as uint * o as uint }
    #[inline]
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

    /// Self is an index, return the stride offset
    #[inline]
    fn stride_offset(index: &(Ix, Ix, Ix), strides: &(Ix, Ix, Ix)) -> int
    {
        let (i, j, k) = *index;
        let (s, t, u) = *strides;
        stride_offset(i, s) + stride_offset(j, t) + stride_offset(k, u)
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
    fn slice(&self) -> &[Ix] { self[] }
    fn slice_mut(&mut self) -> &mut [Ix] { self.as_mut_slice() }
}

/// Construct one-dimensional array shape. Helper function to use where
/// integer literal inference isn't working.
#[inline]
#[deprecated="No longer needed, inference is good enough."]
pub fn d1(a: Ix) -> Ix { a }
/// Construct two-dimensional array shape. Helper function to use where
/// integer literal inference isn't working.
#[inline]
#[deprecated="No longer needed, inference is good enough."]
pub fn d2(a: Ix, b: Ix) -> (Ix, Ix) { (a, b) }
/// Construct three-dimensional array shape. Helper function to use where
/// integer literal inference isn't working.
#[inline]
#[deprecated="No longer needed, inference is good enough."]
pub fn d3(a: Ix, b: Ix, c: Ix) -> (Ix, Ix, Ix) { (a, b, c) }
/// Construct four-dimensional array shape. Helper function to use where
/// integer literal inference isn't working.
#[inline]
#[deprecated="No longer needed, inference is good enough."]
pub fn d4(a: Ix, b: Ix, c: Ix, d: Ix) -> (Ix, Ix, Ix, Ix) { (a, b, c, d) }

/// Helper trait to define a larger-than relation for array shapes:
/// removing one axis from *Self* gives smaller dimension *E*.
pub trait RemoveAxis<E: Dimension> : Dimension {
    fn remove_axis(&self, axis: uint) -> E;
}

macro_rules! impl_shrink(
    ($from:ident $(,$more:ident)*) => (
impl RemoveAxis<($($more),*)> for ($from $(,$more)*)
{
    #[allow(unused_parens)]
    fn remove_axis(&self, axis: uint) -> ($($more),*) {
        let mut tup = ($(0 as $more),*);
        {
            let mut it = tup.slice_mut().iter_mut();
            for (i, &d) in self.slice().iter().enumerate() {
                if i == axis {
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
    )
)

macro_rules! impl_shrink_recursive(
    ($ix:ident) => (impl_shrink!($ix));
    ($ix1:ident $(,$ix:ident)*) => (
        impl_shrink_recursive!($($ix),*)
        impl_shrink!($ix1 $(,$ix)*)
    )
)

// 12 is the maximum number for having the Eq trait from libstd
impl_shrink_recursive!(Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix)

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

