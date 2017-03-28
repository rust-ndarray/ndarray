
use std::marker::PhantomData;

use imp_prelude::*;
use IntoDimension;
use {Producer, Layout, NdIndex};
use Iter;

impl<'a, A, D> Producer for WholeChunks<'a, A, D>
    where D: Dimension,
{
    type Item = ArrayView<'a, A, D>;
    type Elem = A;
    type Dim = D;
    fn raw_dim(&self) -> D {
        self.size.clone()
    }
    fn layout(&self) -> Layout {
        if Dimension::is_contiguous(&self.size, &self.strides) {
            Layout::c()
        } else {
            Layout::none()
        }
    }

    fn as_ptr(&self) -> *mut A {
        self.ptr
    }

    fn contiguous_stride(&self) -> isize {
        let n = self.strides.ndim();
        let s = self.strides[n - 1] as isize;
        s
    }

    unsafe fn as_ref(&self, p: *mut A) -> Self::Item {
        ArrayView::from_shape_ptr(self.chunk.clone().strides(self.inner_strides.clone()), p)
    }

    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A {
        self.ptr.offset(i.index_unchecked(&self.strides))
    }

    fn stride_of(&self, axis: Axis) -> isize {
        self.strides[axis.index()] as isize
    }

    fn split_at(self, _axis: Axis, _index: usize) -> (Self, Self) {
        unimplemented!()
    }
    private_impl!{}
}

#[derive(Debug, Clone)]
pub struct WholeChunks<'a, A: 'a, D> {
    size: D,
    chunk: D,
    strides: D,
    inner_strides: D,
    ptr: *mut A,
    life: PhantomData<&'a A>,
}

/// **Panics** if any chunk dimension is zero<br>
pub fn whole_chunks_of<A, D, E>(a: ArrayView<A, D>, chunk: E) -> WholeChunks<A, D>
    where D: Dimension,
          E: IntoDimension<Dim=D>,
{
    let mut chunk = chunk.into_dimension();
    let mut size = a.raw_dim();
    for (sz, ch) in size.slice_mut().iter_mut().zip(chunk.slice_mut()) {
        assert!(*ch != 0, "Chunk size must not be zero");
        if *ch > *sz { *ch = *sz; }
        *sz /= *ch;
    }
    let mut strides = a.raw_dim();
    for (a, b) in strides.slice_mut().iter_mut().zip(a.strides()) {
        *a = *b as Ix;
    }
    
    let mut mult_strides = strides.clone();
    for (a, &b) in mult_strides.slice_mut().iter_mut().zip(chunk.slice()) {
        *a *= b;
    }

    WholeChunks {
        chunk: chunk,
        inner_strides: strides,
        strides: mult_strides.clone(),
        ptr: a.as_ptr() as _,
        size: size,
        life: PhantomData,
    }
}

impl<'a, A, D> IntoIterator for WholeChunks<'a, A, D>
    where D: Dimension,
          A: 'a,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = WholeChunksIter<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter {
        unsafe {
        WholeChunksIter {
            iter: ArrayView::from_shape_ptr(self.size.strides(self.strides), self.ptr).into_iter(),
            chunk: self.chunk,
            inner_strides: self.inner_strides,
        }
        }
    }
}

pub struct WholeChunksIter<'a, A: 'a, D> {
    iter: Iter<'a, A, D>,
    chunk: D,
    inner_strides: D,
}

impl<'a, A, D> Iterator for WholeChunksIter<'a, A, D>
    where D: Dimension,
{
    type Item = ArrayView<'a, A, D>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|elt| {
            unsafe {
                ArrayView::from_shape_ptr(
                    self.chunk.clone()
                        .strides(self.inner_strides.clone()),
                    elt)
            }
        })
    }
}
