use std::slice;

use super::{Si, Ix, Ixs};
use super::zipsl;

/// Calculate offset from `Ix` stride converting sign properly
#[inline]
pub fn stride_offset(n: Ix, stride: Ix) -> isize
{
    (n as isize) * ((stride as Ixs) as isize)
}

/// Trait for the shape and index types of arrays.
///
/// `unsafe` because of the assumptions in the default methods.
///
/// ***Don't implement this trait, it's internal to the crate and will
/// evolve at will.***
pub unsafe trait Dimension : Clone + Eq {
    /// `SliceArg` is the type which is used to specify slicing for this
    /// dimension.
    ///
    /// For the fixed size dimensions (tuples) it is a fixed size array
    /// of the correct size, which you pass by reference. For the `Vec`
    /// dimension it is a slice.
    ///
    /// - For `Ix`: `[Si; 1]`
    /// - For `(Ix, Ix)`: `[Si; 2]`
    /// - and so on..
    /// - For `Vec<Ix>`: `[Si]`
    ///
    /// The easiest way to create a `&SliceArg` is using the macro
    /// [`s![]`](macro.s!.html).
    type SliceArg: ?Sized + AsRef<[Si]>;
    fn ndim(&self) -> usize;
    fn slice(&self) -> &[Ix] {
        unsafe {
            slice::from_raw_parts(self as *const _ as *const Ix, self.ndim())
        }
    }

    fn slice_mut(&mut self) -> &mut [Ix] {
        unsafe {
            slice::from_raw_parts_mut(self as *mut _ as *mut Ix, self.ndim())
        }
    }

    fn size(&self) -> usize {
        self.slice().iter().fold(1, |s, &a| s * a as usize)
    }

    fn default_strides(&self) -> Self {
        // Compute default array strides
        // Shape (a, b, c) => Give strides (b * c, c, 1)
        let mut strides = self.clone();
        {
            let mut it = strides.slice_mut().iter_mut().rev();
            // Set first element to 1
            for rs in it.by_ref() {
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
        for (&dim, ix) in zipsl(self.slice(), index.slice_mut()).rev()
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
    fn stride_offset(index: &Self, strides: &Self) -> isize
    {
        let mut offset = 0;
        for (&i, &s) in zipsl(index.slice(), strides.slice()) {
            offset += stride_offset(i, s);
        }
        offset
    }

    /// Return stride offset for this dimension and index.
    fn stride_offset_checked(&self, strides: &Self, index: &Self) -> Option<isize>
    {
        let mut offset = 0;
        for ((&d, &i), &s) in zipsl(zipsl(self.slice(),
                                          index.slice()),
                                    strides.slice())
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
    fn do_slices(dim: &mut Self, strides: &mut Self, slices: &Self::SliceArg) -> isize
    {
        let slices = slices.as_ref();
        let mut offset = 0;
        assert!(slices.len() == dim.slice().len());
        for ((dr, sr), &slc) in zipsl(zipsl(dim.slice_mut(),
                                            strides.slice_mut()),
                                      slices)
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
pub fn do_sub<A, D: Dimension>(dims: &mut D, ptr: &mut *mut A, strides: &D,
                               axis: usize, index: Ix)
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


unsafe impl Dimension for () {
    type SliceArg = [Si; 0];
    // empty product is 1 -> size is 1
    #[inline]
    fn ndim(&self) -> usize { 0 }
    fn slice(&self) -> &[Ix] { &[] }
    fn slice_mut(&mut self) -> &mut [Ix] { &mut [] }
}

unsafe impl Dimension for Ix {
    type SliceArg = [Si; 1];
    #[inline]
    fn ndim(&self) -> usize { 1 }
    #[inline]
    fn size(&self) -> usize { *self as usize }

    #[inline]
    fn default_strides(&self) -> Self { 1 }

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
    fn stride_offset(index: &Ix, stride: &Ix) -> isize
    {
        stride_offset(*index, *stride)
    }

    /// Return stride offset for this dimension and index.
    #[inline]
    fn stride_offset_checked(&self, stride: &Ix, index: &Ix) -> Option<isize>
    {
        if *index < *self {
            Some(stride_offset(*index, *stride))
        } else {
            None
        }
    }
}

unsafe impl Dimension for (Ix, Ix) {
    type SliceArg = [Si; 2];
    #[inline]
    fn ndim(&self) -> usize { 2 }
    #[inline]
    fn size(&self) -> usize { let (m, n) = *self; m as usize * n as usize }
    #[inline]
    fn default_strides(&self) -> Self {
        // Compute default array strides
        // Shape (a, b, c) => Give strides (b * c, c, 1)
        (self.1, 1)
    }

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
    fn stride_offset(index: &(Ix, Ix), strides: &(Ix, Ix)) -> isize
    {
        let (i, j) = *index;
        let (s, t) = *strides;
        stride_offset(i, s) + stride_offset(j, t)
    }

    /// Return stride offset for this dimension and index.
    #[inline]
    fn stride_offset_checked(&self, strides: &(Ix, Ix), index: &(Ix, Ix)) -> Option<isize>
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

unsafe impl Dimension for (Ix, Ix, Ix) {
    type SliceArg = [Si; 3];
    #[inline]
    fn ndim(&self) -> usize { 3 }
    #[inline]
    fn size(&self) -> usize { let (m, n, o) = *self; m as usize * n as usize * o as usize }
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
    fn stride_offset(index: &(Ix, Ix, Ix), strides: &(Ix, Ix, Ix)) -> isize
    {
        let (i, j, k) = *index;
        let (s, t, u) = *strides;
        stride_offset(i, s) + stride_offset(j, t) + stride_offset(k, u)
    }
}

macro_rules! large_dim {
    ($n:expr, $($ix:ident),+) => (
        unsafe impl Dimension for ($($ix),+) {
            type SliceArg = [Si; $n];
            #[inline]
            fn ndim(&self) -> usize { $n }
        }
    )
}

large_dim!(4, Ix, Ix, Ix, Ix);
large_dim!(5, Ix, Ix, Ix, Ix, Ix);
large_dim!(6, Ix, Ix, Ix, Ix, Ix, Ix);
large_dim!(7, Ix, Ix, Ix, Ix, Ix, Ix, Ix);
large_dim!(8, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix);
large_dim!(9, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix);
large_dim!(10, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix);
large_dim!(11, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix);
large_dim!(12, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix);

/// Vec<Ix> is a "dynamic" index, pretty hard to use when indexing,
/// and memory wasteful, but it allows an arbitrary and dynamic number of axes.
unsafe impl Dimension for Vec<Ix>
{
    type SliceArg = [Si];
    fn ndim(&self) -> usize { self.len() }
    fn slice(&self) -> &[Ix] { self }
    fn slice_mut(&mut self) -> &mut [Ix] { self }
}

/// Helper trait to define a larger-than relation for array shapes:
/// removing one axis from *Self* gives smaller dimension *Smaller*.
pub trait RemoveAxis : Dimension {
    type Smaller: Dimension;
    fn remove_axis(&self, axis: usize) -> Self::Smaller;
}

macro_rules! impl_shrink(
    ($from:ident, $($more:ident,)*) => (
impl RemoveAxis for ($from $(,$more)*)
{
    type Smaller = ($($more),*);
    #[allow(unused_parens)]
    #[inline]
    fn remove_axis(&self, axis: usize) -> ($($more),*) {
        let mut tup = ($(0 as $more),*);
        {
            let mut it = tup.slice_mut().iter_mut();
            for (i, &d) in self.slice().iter().enumerate() {
                if i == axis {
                    continue;
                }
                for rr in it.by_ref() {
                    *rr = d;
                    break
                }
            }
        }
        tup
    }
}
    )
);

macro_rules! impl_shrink_recursive(
    ($ix:ident, ) => (impl_shrink!($ix,););
    ($ix1:ident, $($ix:ident,)*) => (
        impl_shrink_recursive!($($ix,)*);
        impl_shrink!($ix1, $($ix,)*);
    )
);

// 12 is the maximum number for having the Eq trait from libstd
impl_shrink_recursive!(Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix, Ix,);

impl RemoveAxis for Vec<Ix>
{
    type Smaller = Vec<Ix>;
    fn remove_axis(&self, axis: usize) -> Vec<Ix>
    {
        let mut res = self.clone();
        res.remove(axis);
        res
    }
}

/// A tuple or fixed size array that can be used to index an array.
///
/// ```
/// use ndarray::arr2;
///
/// let mut a = arr2(&[[0, 1], [0, 0]]);
/// a[[1, 1]] = 1;
/// assert_eq!(a[[0, 1]], 1);
/// assert_eq!(a[[1, 1]], 1);
/// ```
///
/// **Note** the blanket implementation that's not visible in rustdoc:
/// `impl<D> NdIndex for D where D: Dimension { ... }`
pub unsafe trait NdIndex {
    type Dim: Dimension;
    #[doc(hidden)]
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize>;
}

unsafe impl<D> NdIndex for D where D: Dimension {
    type Dim = D;
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize> {
        dim.stride_offset_checked(strides, self)
    }
}

unsafe impl NdIndex for [Ix; 0] {
    type Dim = ();
    #[inline]
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize> {
        dim.stride_offset_checked(strides, &())
    }
}

unsafe impl NdIndex for [Ix; 1] {
    type Dim = Ix;
    #[inline]
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize> {
        dim.stride_offset_checked(strides, &self[0])
    }
}

unsafe impl NdIndex for [Ix; 2] {
    type Dim = (Ix, Ix);
    #[inline]
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize> {
        let index = (self[0], self[1]);
        dim.stride_offset_checked(strides, &index)
    }
}

unsafe impl NdIndex for [Ix; 3] {
    type Dim = (Ix, Ix, Ix);
    #[inline]
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize> {
        let index = (self[0], self[1], self[2]);
        dim.stride_offset_checked(strides, &index)
    }
}

unsafe impl NdIndex for [Ix; 4] {
    type Dim = (Ix, Ix, Ix, Ix);
    #[inline]
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize> {
        let index = (self[0], self[1], self[2], self[3]);
        dim.stride_offset_checked(strides, &index)
    }
}

unsafe impl<'a> NdIndex for &'a [Ix] {
    type Dim = Vec<Ix>;
    fn index_checked(&self, dim: &Self::Dim, strides: &Self::Dim) -> Option<isize> {
        let mut offset = 0;
        for ((&d, &i), &s) in zipsl(zipsl(&dim[..], &self[..]),
                                    strides.slice())
        {
            if i >= d {
                return None;
            }
            offset += stride_offset(i, s);
        }
        Some(offset)
    }
}
