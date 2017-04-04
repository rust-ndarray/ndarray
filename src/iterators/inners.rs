
use imp_prelude::*;
use {NdProducer, Layout};
use super::InnerIter;
use super::InnerIterMut;

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone ]
    Inners {
        base,
        inner_len,
        inner_stride,
    }
    Inners<'a, A, D> {
        type Dim = D;
        type Item = ArrayView<'a, A, Ix1>;

        unsafe fn item(&self, ptr) {
            ArrayView::new_(ptr, Ix1(self.inner_len), Ix1(self.inner_stride as Ix))
        }
    }
}

/// See [`.inners()`](struct.ArrayBase.html#method.inners)
/// for more information.
pub struct Inners<'a, A: 'a, D> {
    base: ArrayView<'a, A, D>,
    inner_len: Ix,
    inner_stride: Ixs,
}


pub fn new_inners<A, D>(v: ArrayView<A, D>, axis: Axis)
    -> Inners<A, D::Smaller>
    where D: Dimension
{
    let ndim = v.ndim();
    let len;
    let stride;
    let iter_v;
    if ndim == 0 {
        len = 1;
        stride = 1;
        iter_v = v.try_remove_axis(Axis(0))
    } else {
        let i = axis.index();
        len = v.dim[i];
        stride = v.strides[i] as isize;
        iter_v = v.try_remove_axis(axis)
    }
    Inners {
        inner_len: len,
        inner_stride: stride,
        base: iter_v,
    }
}

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone =>]
    InnersMut {
        base,
        inner_len,
        inner_stride,
    }
    InnersMut<'a, A, D> {
        type Dim = D;
        type Item = ArrayViewMut<'a, A, Ix1>;

        unsafe fn item(&self, ptr) {
            ArrayViewMut::new_(ptr, Ix1(self.inner_len), Ix1(self.inner_stride as Ix))
        }
    }
}

impl<'a, A, D> IntoIterator for Inners<'a, A, D>
    where D: Dimension,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = InnerIter<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter {
        InnerIter {
            iter: self.base.into_base_iter(),
            inner_len: self.inner_len,
            inner_stride: self.inner_stride,
        }
    }
}

/// See [`.inners_mut()`](struct.ArrayBase.html#method.inners_mut)
/// for more information.
pub struct InnersMut<'a, A: 'a, D> {
    base: ArrayViewMut<'a, A, D>,
    inner_len: Ix,
    inner_stride: Ixs,
}


pub fn new_inners_mut<A, D>(v: ArrayViewMut<A, D>, axis: Axis)
    -> InnersMut<A, D::Smaller>
    where D: Dimension
{
    let ndim = v.ndim();
    let len;
    let stride;
    let iter_v;
    if ndim == 0 {
        len = 1;
        stride = 1;
        iter_v = v.try_remove_axis(Axis(0))
    } else {
        let i = axis.index();
        len = v.dim[i];
        stride = v.strides[i] as isize;
        iter_v = v.try_remove_axis(axis)
    }
    InnersMut {
        inner_len: len,
        inner_stride: stride,
        base: iter_v,
    }
}

impl<'a, A, D> IntoIterator for InnersMut<'a, A, D>
    where D: Dimension,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = InnerIterMut<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter {
        InnerIterMut {
            iter: self.base.into_base_iter(),
            inner_len: self.inner_len,
            inner_stride: self.inner_stride,
        }
    }
}
