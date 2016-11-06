
use std::fmt::Debug;

use itertools::zip;

use {Ix, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, IxDyn, Dim, Dimension, IntoDimension};
use {zipsl, ZipExt};
use super::{stride_offset};
use super::DimPrivate;

/// Tuple or fixed size arrays that can be used to index an array.
///
/// ```
/// use ndarray::arr2;
///
/// let mut a = arr2(&[[0, 1],
///                    [2, 3]]);
/// assert_eq!(a[[0, 1]], 1);
/// assert_eq!(a[[1, 1]], 3);
/// a[[1, 1]] += 1;
/// assert_eq!(a[(1, 1)], 4);
/// ```
///
/// **Note** that `NdIndex` is implemented for all `D where D: Dimension`.
pub unsafe trait NdIndex<E> : Debug {
    #[doc(hidden)]
    fn index_checked(&self, dim: &E, strides: &E) -> Option<isize>;
    fn index_unchecked(&self, strides: &E) -> isize;
}

unsafe impl<D> NdIndex<D> for D
    where D: Dimension
{
    fn index_checked(&self, dim: &D, strides: &D) -> Option<isize> {
        dim.stride_offset_checked(strides, self)
    }
    fn index_unchecked(&self, strides: &D) -> isize {
        D::stride_offset(self, strides)
    }
}

unsafe impl NdIndex<Ix0> for () {
    #[inline]
    fn index_checked(&self, dim: &Ix0, strides: &Ix0) -> Option<isize> {
        dim.stride_offset_checked(strides, &Ix0())
    }
    #[inline(always)]
    fn index_unchecked(&self, _strides: &Ix0) -> isize {
        0
    }
}

unsafe impl NdIndex<Ix1> for Ix {
    #[inline]
    fn index_checked(&self, dim: &Ix1, strides: &Ix1) -> Option<isize> {
        dim.stride_offset_checked(strides, &Ix1(*self))
    }
    #[inline(always)]
    fn index_unchecked(&self, strides: &Ix1) -> isize {
        stride_offset(*self, get!(strides, 0))
    }
}

unsafe impl NdIndex<Ix2> for (Ix, Ix) {
    #[inline]
    fn index_checked(&self, dim: &Ix2, strides: &Ix2) -> Option<isize> {
        dim.stride_offset_checked(strides, &Ix2(self.0, self.1))
    }
    #[inline]
    fn index_unchecked(&self, strides: &Ix2) -> isize {
        stride_offset(self.0, get!(strides, 0)) + 
        stride_offset(self.1, get!(strides, 1))
    }
}
unsafe impl NdIndex<Ix3> for (Ix, Ix, Ix) {
    #[inline]
    fn index_checked(&self, dim: &Ix3, strides: &Ix3) -> Option<isize> {
        dim.stride_offset_checked(strides, &self.into_dimension())
    }

    #[inline]
    fn index_unchecked(&self, strides: &Ix3) -> isize {
        stride_offset(self.0, get!(strides, 0)) + 
        stride_offset(self.1, get!(strides, 1)) +
        stride_offset(self.2, get!(strides, 2))
    }
}

unsafe impl NdIndex<Ix4> for (Ix, Ix, Ix, Ix) {
    #[inline]
    fn index_checked(&self, dim: &Ix4, strides: &Ix4) -> Option<isize> {
        dim.stride_offset_checked(strides, &self.into_dimension())
    }
    #[inline]
    fn index_unchecked(&self, strides: &Ix4) -> isize {
        zip(strides.ix(), self.into_dimension().ix()).map(|(&s, &i)| stride_offset(i, s)).sum()
    }
}
unsafe impl NdIndex<Ix5> for (Ix, Ix, Ix, Ix, Ix) {
    #[inline]
    fn index_checked(&self, dim: &Ix5, strides: &Ix5) -> Option<isize> {
        dim.stride_offset_checked(strides, &self.into_dimension())
    }
    #[inline]
    fn index_unchecked(&self, strides: &Ix5) -> isize {
        zip(strides.ix(), self.into_dimension().ix()).map(|(&s, &i)| stride_offset(i, s)).sum()
    }
}

unsafe impl NdIndex<Ix2> for [Ix; 2] {
    #[inline]
    fn index_checked(&self, dim: &Ix2, strides: &Ix2) -> Option<isize> {
        dim.stride_offset_checked(strides, &Ix2(self[0], self[1]))
    }
    #[inline]
    fn index_unchecked(&self, strides: &Ix2) -> isize {
        stride_offset(self[0], get!(strides, 0)) + 
        stride_offset(self[1], get!(strides, 1))
    }
}

unsafe impl NdIndex<Ix3> for [Ix; 3] {
    #[inline]
    fn index_checked(&self, dim: &Ix3, strides: &Ix3) -> Option<isize> {
        dim.stride_offset_checked(strides, &Ix3(self[0], self[1], self[2]))
    }
    #[inline]
    fn index_unchecked(&self, strides: &Ix3) -> isize {
        stride_offset(self[0], get!(strides, 0)) + 
        stride_offset(self[1], get!(strides, 1)) +
        stride_offset(self[2], get!(strides, 2))
    }
}

impl<'a> IntoDimension for &'a [Ix] {
    type Dim = Dim<Vec<Ix>>;
    fn into_dimension(self) -> Self::Dim {
        Dim(self.to_vec())
    }
}

unsafe impl<'a> NdIndex<IxDyn> for &'a [Ix] {
    fn index_checked(&self, dim: &IxDyn, strides: &IxDyn) -> Option<isize> {
        let mut offset = 0;
        for (&d, &i, &s) in zipsl(&dim[..], &self[..]).zip_cons(strides.slice()) {
            if i >= d {
                return None;
            }
            offset += stride_offset(i, s);
        }
        Some(offset)
    }
    fn index_unchecked(&self, strides: &IxDyn) -> isize {
        zip(strides.ix(), *self).map(|(&s, &i)| stride_offset(i, s)).sum()
    }
}

unsafe impl NdIndex<IxDyn> for Vec<Ix> {
    fn index_checked(&self, dim: &IxDyn, strides: &IxDyn) -> Option<isize> {
        let mut offset = 0;
        for (&d, &i, &s) in zipsl(&dim[..], &self[..]).zip_cons(strides.slice()) {
            if i >= d {
                return None;
            }
            offset += stride_offset(i, s);
        }
        Some(offset)
    }
    fn index_unchecked(&self, strides: &IxDyn) -> isize {
        zip(strides.ix(), self).map(|(&s, &i)| stride_offset(i, s)).sum()
    }
}
