
#[macro_use]
extern crate ndarray;

use ndarray::prelude::*;
use ndarray::IntoDimension;
use ndarray::{Offset, Producer, Layout, NdIndex};
use ndarray::Zip;
use std::marker::PhantomData;

impl<'a, A, D> Producer for Chunks<'a, A, D>
    where D: Dimension,
{
    type Ref = ArrayView<'a, A, D>;
    type Ptr = *mut A;
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

    fn as_ptr(&self) -> Self::Ptr {
        self.ptr
    }

    fn contiguous_stride(&self) -> isize {
        let n = self.strides.ndim();
        let s = self.strides[n - 1] as isize;
        s
    }

    unsafe fn as_ref(&self, p: Self::Ptr) -> Self::Ref {
        ArrayView::from_shape_ptr(self.chunk.clone().strides(self.inner_strides.clone()), p)
    }

    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
        let n = self.strides.ndim();
        let s = self.strides[n - 1] as isize;
        self.ptr.offset(i.index_unchecked(&self.strides))
    }

    fn stride_of(&self, axis: Axis) -> isize {
        self.strides[axis.index()] as isize
    }

    fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
        panic!()
    }
    private_impl!{}
}

#[derive(Debug, Clone)]
pub struct Chunks<'a, A: 'a, D> {
    size: D,
    chunk: D,
    strides: D,
    inner_strides: D,
    ptr: *mut A,
    life: PhantomData<&'a A>,
}

/// **Panics** if any chunk dimension is zero<br>
pub fn chunks<A, D, E>(a: ArrayView<A, D>, chunk: E) -> Chunks<A, D>
    where D: Dimension,
          E: IntoDimension<Dim=D>,
{
    let chunk = chunk.into_dimension();
    let mut size = a.raw_dim();
    for (sz, &ch) in size.slice_mut().iter_mut().zip(chunk.slice()) {
        assert!(ch != 0, "Chunk size must not be zero");
        let mut d = *sz / ch;
        *sz = d;
    }
    let mut strides = a.raw_dim();
    for (a, b) in strides.slice_mut().iter_mut().zip(a.strides()) {
        *a = *b as Ix;
    }
    
    let mut mult_strides = strides.clone();
    for (a, &b) in mult_strides.slice_mut().iter_mut().zip(chunk.slice()) {
        *a *= b;
    }

    Chunks {
        chunk: chunk,
        inner_strides: strides,
        strides: mult_strides.clone(),
        ptr: a.as_ptr() as _,
        size: size,
        life: PhantomData,
    }
}

impl<'a, A, D> IntoIterator for Chunks<'a, A, D>
    where D: Dimension,
          A: 'a,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = ChunkIter<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter {
        unsafe {
        ChunkIter {
            iter: ArrayView::from_shape_ptr(self.size.strides(self.strides), self.ptr).into_iter(),
            chunk: self.chunk,
            inner_strides: self.inner_strides,
        }
        }
    }
}

pub struct ChunkIter<'a, A: 'a, D> {
    iter: ndarray::Iter<'a, A, D>,
    chunk: D,
    inner_strides: D,
}

impl<'a, A, D> Iterator for ChunkIter<'a, A, D>
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

fn main() {
    let mut a = <Array1<f32>>::linspace(1., 100., 10 * 10).into_shape((10, 10)).unwrap();
    let iter = chunks(a.t(), Dim((5, 4)));
    for elt in iter {
        println!("{:6.2?}", elt);
    }
    let iter = chunks(a.t(), Dim((5, 4)));
    /*
    */
    let mut b = <Array2<f32>>::zeros((2, 2));
    println!("{:?}", 
        Zip::from(&mut b).and_view(iter.clone())
    );

    Zip::from(&mut b).and_view(iter).apply(|b, a| {
        println!("{:6.2?}", a);
        *b = a.row(0).scalar_sum();
    });
    println!("{:?}", b);
    Zip::from(b.view_mut().reversed_axes()).and_view(chunks(a.view(), Dim([4, 5]))).apply(|b, a| {
        println!("{:6.2?}", a);
        *b = a.row(0).scalar_sum();
    });
    println!("{:?}", b);
    //array_zip!(mut a (a), mut b in { *b = a.scalar_sum() });
}
