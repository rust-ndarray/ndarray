// Copyright 2017 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_use]
mod zipmacro;

use imp_prelude::*;
use IntoDimension;
use NdIndex;
use Layout;

use layout::{CORDER, FORDER};
use layout::LayoutPriv;
use indexes::{Indices, indices};

/// Return if the expression is a break value.
macro_rules! fold_while {
    ($e:expr) => {
        match $e {
            FoldWhile::Continue(x) => x,
            x => return x,
        }
    }
}


/// Broadcast an array so that it acts like a larger size and/or shape array.
///
/// See [broadcasting][1] for more information.
///
/// [1]: struct.ArrayBase.html#broadcasting
trait Broadcast<E>
    where E: IntoDimension,
{
    type Output: NdProducer<Dim=E::Dim>;
    /// Broadcast the array to the new dimensions `shape`.
    ///
    /// ***Panics*** if broadcasting isn’t possible.
    fn broadcast_unwrap(self, shape: E) -> Self::Output;
    private_decl!{}
}

impl<S, D> ArrayBase<S, D>
    where S: Data,
          D: Dimension,
{
    pub(crate) fn layout_impl(&self) -> Layout {
        Layout::new(if self.is_standard_layout() {
            if self.ndim() <= 1 {
                FORDER | CORDER
            } else {
                CORDER
            }
        } else if self.ndim() > 1 && self.t().is_standard_layout() {
            FORDER
        } else {
            0
        })
    }
}

impl<'a, A, D, E> Broadcast<E> for ArrayView<'a, A, D>
    where E: IntoDimension,
          D: Dimension,
{
    type Output = ArrayView<'a, A, E::Dim>;
    fn broadcast_unwrap(self, shape: E) -> Self::Output {
        let res: ArrayView<A, E::Dim> = (&self).broadcast_unwrap(shape.into_dimension());
        unsafe {
            ArrayView::new_(res.ptr, res.dim, res.strides)
        }
    }
    private_impl!{}
}

pub trait Splittable : Sized {
    fn split_at(self, Axis, Ix) -> (Self, Self);
}

impl<D> Splittable for D
    where D: Dimension,
{
    fn split_at(self, axis: Axis, index: Ix) -> (Self, Self) {
        let mut d1 = self;
        let mut d2 = d1.clone();
        let i = axis.index();
        let len = d1[i];
        d1[i] = index;
        d2[i] = len - index;
        (d1, d2)
    }
}

/// Argument conversion into a producer.
///
/// Slices and vectors can be used (equivalent to 1-dimensional array views).
///
/// This trait is like `IntoIterator` for `NdProducers` instead of iterators.
pub trait IntoNdProducer {
    /// The element produced per iteration.
    type Item;
    /// Dimension type of the producer
    type Dim: Dimension;
    type Output: NdProducer<Dim=Self::Dim, Item=Self::Item>;
    /// Convert the value into an `NdProducer`.
    fn into_producer(self) -> Self::Output;
}

impl<P> IntoNdProducer for P where P: NdProducer {
    type Item = P::Item;
    type Dim = P::Dim;
    type Output = Self;
    fn into_producer(self) -> Self::Output { self }
}

/// A producer of an n-dimensional set of elements;
/// for example an array view, mutable array view or an iterator
/// that yields chunks.
///
/// Producers are used as a arguments to `Zip` and `azip!()`.
///
/// # Comparison to `IntoIterator`
///
/// Most `NdProducers` are *iterable* (implement `IntoIterator`) but not directly
/// iterators. This separation is needed because the producer represents
/// a multidimensional set of items, it can be split along a particular axis for
/// parallelization, and it has no fixed correspondance to a sequence.
///
/// The natural exception is one dimensional producers, like `AxisIter`, which
/// implement `Iterator` directly
/// (`AxisIter` traverses a one dimensional sequence, along an axis, while
/// *producing* multidimensional items).
///
/// See also [`IntoNdProducer`](trait.IntoNdProducer.html)
pub trait NdProducer {
    /// The element produced per iteration.
    type Item;
    // Internal use / Pointee type
    /// Dimension type
    type Dim: Dimension;

    // The pointer Ptr is used by an array view to simply point to the
    // current element. It doesn't have to be a pointer (see Indices).
    // Its main function is that it can be incremented with a particular
    // stride (= along a particular axis)
    #[doc(hidden)]
    /// Pointer or stand-in for pointer
    type Ptr: Offset<Stride=Self::Stride>;
    #[doc(hidden)]
    /// Pointer stride
    type Stride: Copy;

    #[doc(hidden)]
    fn layout(&self) -> Layout;
    #[doc(hidden)]
    fn raw_dim(&self) -> Self::Dim;
    #[doc(hidden)]
    fn equal_dim(&self, dim: &Self::Dim) -> bool {
        self.raw_dim() == *dim
    }
    #[doc(hidden)]
    fn as_ptr(&self) -> Self::Ptr;
    #[doc(hidden)]
    unsafe fn as_ref(&self, Self::Ptr) -> Self::Item;
    #[doc(hidden)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr;
    #[doc(hidden)]
    fn stride_of(&self, axis: Axis) -> <Self::Ptr as Offset>::Stride;
    #[doc(hidden)]
    #[inline(always)]
    fn contiguous_stride(&self) -> Self::Stride;
    #[doc(hidden)]
    fn split_at(self, axis: Axis, index: usize) -> (Self, Self) where Self: Sized;
    private_decl!{}
}

pub trait Offset : Copy {
    type Stride: Copy;
    unsafe fn stride_offset(self, s: Self::Stride, index: usize) -> Self;
    private_decl!{}
}

impl<T> Offset for *mut T {
    type Stride = isize;
    unsafe fn stride_offset(self, s: Self::Stride, index: usize) -> Self {
        self.offset(s * (index as isize))
    }
    private_impl!{}
}

trait ZippableTuple : Sized {
    type Item;
    type Ptr: OffsetTuple<Args=Self::Stride> + Copy;
    type Dim: Dimension;
    type Stride: Copy;
    fn as_ptr(&self) -> Self::Ptr;
    unsafe fn as_ref(&self, Self::Ptr) -> Self::Item;
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr;
    fn stride_of(&self, index: usize) -> Self::Stride;
    fn contiguous_stride(&self) -> Self::Stride;
    fn split_at(self, axis: Axis, index: usize) -> (Self, Self);
}

/// An array reference is an n-dimensional producer of element references
/// (like ArrayView).
impl<'a, A: 'a, S, D> IntoNdProducer for &'a ArrayBase<S, D>
    where D: Dimension,
          S: Data<Elem=A>,
{
    type Item = &'a A;
    type Dim = D;
    type Output = ArrayView<'a, A, D>;
    fn into_producer(self) -> Self::Output {
        self.view()
    }
}

/// A mutable array reference is an n-dimensional producer of mutable element
/// references (like ArrayViewMut).
impl<'a, A: 'a, S, D> IntoNdProducer for &'a mut ArrayBase<S, D>
    where D: Dimension,
          S: DataMut<Elem=A>,
{
    type Item = &'a mut A;
    type Dim = D;
    type Output = ArrayViewMut<'a, A, D>;
    fn into_producer(self) -> Self::Output {
        self.view_mut()
    }
}

/// A slice is a one-dimensional producer
impl<'a, A: 'a> IntoNdProducer for &'a [A] {
    type Item = <Self::Output as NdProducer>::Item;
    type Dim = Ix1;
    type Output = ArrayView1<'a, A>;
    fn into_producer(self) -> Self::Output {
        <_>::from(self)
    }
}

/// A mutable slice is a mutable one-dimensional producer
impl<'a, A: 'a> IntoNdProducer for &'a mut [A] {
    type Item = <Self::Output as NdProducer>::Item;
    type Dim = Ix1;
    type Output = ArrayViewMut1<'a, A>;
    fn into_producer(self) -> Self::Output {
        <_>::from(self)
    }
}

/// A Vec is a one-dimensional producer
impl<'a, A: 'a> IntoNdProducer for &'a Vec<A> {
    type Item = <Self::Output as NdProducer>::Item;
    type Dim = Ix1;
    type Output = ArrayView1<'a, A>;
    fn into_producer(self) -> Self::Output {
        <_>::from(self)
    }
}

/// A mutable Vec is a mutable one-dimensional producer
impl<'a, A: 'a> IntoNdProducer for &'a mut Vec<A> {
    type Item = <Self::Output as NdProducer>::Item;
    type Dim = Ix1;
    type Output = ArrayViewMut1<'a, A>;
    fn into_producer(self) -> Self::Output {
        <_>::from(self)
    }
}

impl<'a, A, D: Dimension> NdProducer for ArrayView<'a, A, D>
{
    type Item = &'a A;
    type Dim = D;
    type Ptr = *mut A;
    type Stride = isize;

    private_impl!{}
    #[doc(hidden)]
    fn raw_dim(&self) -> Self::Dim {
        self.raw_dim()
    }

    #[doc(hidden)]
    fn equal_dim(&self, dim: &Self::Dim) -> bool {
        self.dim.equal(dim)
    }

    #[doc(hidden)]
    fn as_ptr(&self) -> *mut A {
        self.as_ptr() as _
    }

    #[doc(hidden)]
    fn layout(&self) -> Layout {
        self.layout_impl()
    }

    #[doc(hidden)]
    unsafe fn as_ref(&self, ptr: *mut A) -> Self::Item {
        &*ptr
    }

    #[doc(hidden)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A {
        self.ptr.offset(i.index_unchecked(&self.strides))
    }

    #[doc(hidden)]
    fn stride_of(&self, axis: Axis) -> isize {
        self.strides()[axis.index()]
    }

    #[inline(always)]
    fn contiguous_stride(&self) -> Self::Stride { 1 }

    #[doc(hidden)]
    fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
        self.split_at(axis, index)
    }
}

impl<'a, A, D: Dimension> NdProducer for ArrayViewMut<'a, A, D> {
    type Item = &'a mut A;
    type Dim = D;
    type Ptr = *mut A;
    type Stride = isize;

    private_impl!{}
    #[doc(hidden)]
    fn raw_dim(&self) -> Self::Dim {
        self.raw_dim()
    }

    #[doc(hidden)]
    fn equal_dim(&self, dim: &Self::Dim) -> bool {
        self.dim.equal(dim)
    }

    #[doc(hidden)]
    fn as_ptr(&self) -> *mut A {
        self.as_ptr() as _
    }

    #[doc(hidden)]
    fn layout(&self) -> Layout {
        self.layout_impl()
    }

    #[doc(hidden)]
    unsafe fn as_ref(&self, ptr: *mut A) -> Self::Item {
        &mut *ptr
    }

    #[doc(hidden)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A {
        self.ptr.offset(i.index_unchecked(&self.strides))
    }

    #[doc(hidden)]
    fn stride_of(&self, axis: Axis) -> isize {
        self.strides()[axis.index()]
    }

    #[inline(always)]
    fn contiguous_stride(&self) -> Self::Stride { 1 }
    
    #[doc(hidden)]
    fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
        self.split_at(axis, index)
    }
}


/// Lock step function application across several arrays or other producers.
///
/// Zip allows matching several producers to each other elementwise and applying
/// a function over all tuples of elements (one item from each input at
/// a time).
///
/// In general, the zip uses a tuple of producers
/// ([`NdProducer`](trait.NdProducer.html) trait) that all have to be of the
/// same shape. The NdProducer implementation defines what its item type is
/// (for example if it's a shared reference, mutable reference or an array
/// view etc).
///
/// If all the input arrays are of the same memory layout the zip performs much
/// better and the compiler can usually vectorize the loop (if applicable).
///
/// The order elements are visited is not specified. The producers don’t have to
/// have the same item type.
///
/// The `Zip` has two methods for function application: `apply` and
/// `fold_while`. The zip object can be split, which allows parallelization.
/// A read-only zip object (no mutable producers) can be cloned.
///
/// See also the [`azip!()` macro][az] which offers a convenient shorthand
/// to common ways to use `Zip`.
///
/// [az]: macro.azip.html
///
/// ```
/// use ndarray::Zip;
/// use ndarray::Array2;
///
/// type M = Array2<f64>;
///
/// // Create four 2d arrays of the same size
/// let mut a = M::zeros((64, 32));
/// let b = M::from_elem(a.dim(), 1.);
/// let c = M::from_elem(a.dim(), 2.);
/// let d = M::from_elem(a.dim(), 3.);
///
/// // Example 1: Perform an elementwise arithmetic operation across
/// // the four arrays a, b, c, d.
///
/// Zip::from(&mut a)
///     .and(&b)
///     .and(&c)
///     .and(&d)
///     .apply(|w, &x, &y, &z| {
///         *w += x + y * z;
///     });
///
/// // Example 2: Create a new array `e` with one entry per row of `a`.
/// //  Use Zip to traverse the rows of `a` and assign to the corresponding
/// //  entry in `e` with the sum across each row.
/// //  This is possible because the producer for `e` and the row producer
/// //  for `a` have the same shape and dimensionality.
/// //  The rows producer yields one array view (`row`) per iteration.
///
/// use ndarray::{Array1, Axis};
///
/// let mut e = Array1::zeros(a.rows());
///
/// Zip::from(&mut e)
///     .and(a.genrows())
///     .apply(|e, row| {
///         *e = row.scalar_sum();
///     });
///
/// // Check the result against the built in `.sum_axis()` along axis 1.
/// assert_eq!(e, a.sum_axis(Axis(1)));
/// ```
#[derive(Debug, Clone)]
pub struct Zip<Parts, D> {
    parts: Parts,
    dimension: D,
    layout: Layout,
}

impl<P, D> Zip<(P, ), D>
    where D: Dimension,
          P: NdProducer<Dim=D>
{
    /// Create a new `Zip` from the input array or other producer `p`.
    ///
    /// The Zip will take the exact dimension of `p` and all inputs
    /// must have the same dimensions (or be broadcast to them).
    pub fn from<IP>(p: IP) -> Self
        where IP: IntoNdProducer<Dim=D, Output=P, Item=P::Item>
    {
        let array = p.into_producer();
        let dim = array.raw_dim();
        Zip {
            dimension: dim,
            layout: array.layout(),
            parts: (array, ),
        }
    }
}
impl<P, D> Zip<(Indices<D>, P), D>
    where D: Dimension + Copy,
          P: NdProducer<Dim=D>,
{
    /// Create a new `Zip` with an index producer and the producer `p`.
    ///
    /// The Zip will take the exact dimension of `p` and all inputs
    /// must have the same dimensions (or be broadcast to them).
    ///
    /// *Note:* Indexed zip has overhead.
    pub fn indexed<IP>(p: IP) -> Self
        where IP: IntoNdProducer<Dim=D, Output=P, Item=P::Item>
    {
        let array = p.into_producer();
        let dim = array.raw_dim();
        Zip::from(indices(dim)).and(array)
    }
}

impl<Parts, D> Zip<Parts, D>
    where D: Dimension,
{

    fn check<P>(&self, part: &P)
        where P: NdProducer<Dim=D>
    {
        ndassert!(part.equal_dim(&self.dimension),
            "Zip: Producer dimension mismatch, expected: {:?}, got: {:?}",
            self.dimension, part.raw_dim());
    }

    /// Return a the number of element tuples in the Zip
    pub fn size(&self) -> usize {
        self.dimension.size()
    }

    /// Return the length of `axis`
    ///
    /// ***Panics*** if `axis` is out of bounds.
    fn len_of(&self, axis: Axis) -> usize {
        self.dimension[axis.index()]
    }

    /// Return an *approximation* to the max stride axis; if
    /// component arrays disagree, there may be no choice better than the
    /// others.
    fn max_stride_axis(&self) -> Axis {
        let i = match self.layout.flag() {
            FORDER => self.dimension.slice().iter()
                          .rposition(|&len| len > 1).unwrap_or(self.dimension.ndim() - 1),
            /* corder or default */
            _ => self.dimension.slice().iter()
                          .position(|&len| len > 1).unwrap_or(0),
        };
        Axis(i)
    }

}

impl<P, D> Zip<P, D>
    where D: Dimension,
{
    fn apply_core<F, Acc>(&mut self, acc: Acc, function: F) -> FoldWhile<Acc>
        where F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
              P: ZippableTuple<Dim=D>,
    {
        if self.layout.is(CORDER | FORDER) {
            self.apply_core_contiguous(acc, function)
        } else {
            self.apply_core_strided(acc, function)
        }
    }
    fn apply_core_contiguous<F, Acc>(&mut self, mut acc: Acc, mut function: F) -> FoldWhile<Acc>
        where F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
              P: ZippableTuple<Dim=D>,
    {
        debug_assert!(self.layout.is(CORDER | FORDER));
        let size = self.dimension.size();
        let ptrs = self.parts.as_ptr();
        let inner_strides = self.parts.contiguous_stride();
        for i in 0..size {
            unsafe {
                let ptr_i = ptrs.stride_offset(inner_strides, i);
                acc = fold_while![function(acc, self.parts.as_ref(ptr_i))];
            }
        }
        FoldWhile::Continue(acc)
    }

    fn apply_core_strided<F, Acc>(&mut self, mut acc: Acc, mut function: F) -> FoldWhile<Acc>
        where F: FnMut(Acc, P::Item) -> FoldWhile<Acc>,
              P: ZippableTuple<Dim=D>,
    {
        let n = self.dimension.ndim();
        if n == 0 {
            panic!("Unreachable: ndim == 0 is contiguous")
        }
        let unroll_axis = n - 1;
        let inner_len = self.dimension[unroll_axis];
        self.dimension[unroll_axis] = 1;
        let mut index_ = self.dimension.first_index();
        let inner_strides = self.parts.stride_of(unroll_axis);
        while let Some(index) = index_ {
            // Let's “unroll” the loop over the innermost axis
            unsafe {
                let ptr = self.parts.uget_ptr(&index);
                for i in 0..inner_len {
                    let p = ptr.stride_offset(inner_strides, i);
                    acc = fold_while!(function(acc, self.parts.as_ref(p)));
                }
            }

            index_ = self.dimension.next_for(index);
        }
        self.dimension[unroll_axis] = inner_len;
        FoldWhile::Continue(acc)
    }
}

/*
trait Offset : Copy {
    unsafe fn offset(self, off: isize) -> Self;
    unsafe fn stride_offset(self, index: usize, stride: isize) -> Self {
        self.offset(index as isize * stride)
    }
}

impl<T> Offset for *mut T {
    unsafe fn offset(self, off: isize) -> Self {
        self.offset(off)
    }
}
*/


trait OffsetTuple {
    type Args;
    unsafe fn stride_offset(self, stride: Self::Args, index: usize) -> Self;
}

impl<T> OffsetTuple for *mut T {
    type Args = isize;
    unsafe fn stride_offset(self, stride: Self::Args, index: usize) -> Self {
        self.offset(index as isize * stride)
    }
}

macro_rules! offset_impl {
    ($([$($param:ident)*][ $($q:ident)*],)+) => {
        $(
        #[allow(non_snake_case)]
        impl<$($param: Offset),*> OffsetTuple for ($($param, )*) {
            type Args = ($($param::Stride,)*);
            unsafe fn stride_offset(self, stride: Self::Args, index: usize) -> Self {
                let ($($param, )*) = self;
                let ($($q, )*) = stride;
                ($(Offset::stride_offset($param, $q, index),)*)
            }
        }
        )+
    }
}

offset_impl!{
    [A ][ a],
    [A B][ a b],
    [A B C][ a b c],
    [A B C D][ a b c d],
    [A B C D E][ a b c d e],
    [A B C D E F][ a b c d e f],
}

macro_rules! zipt_impl {
    ($([$($p:ident)*][ $($q:ident)*],)+) => {
        $(
        #[allow(non_snake_case)]
        impl<Dim: Dimension, $($p: NdProducer<Dim=Dim>),*> ZippableTuple for ($($p, )*) {
            type Item = ($($p::Item, )*);
            type Ptr = ($($p::Ptr, )*);
            type Dim = Dim;
            type Stride = ($($p::Stride,)* );

            fn stride_of(&self, index: usize) -> Self::Stride {
                let ($(ref $p,)*) = *self;
                ($($p.stride_of(Axis(index)), )*)
            }

            fn contiguous_stride(&self) -> Self::Stride {
                let ($(ref $p,)*) = *self;
                ($($p.contiguous_stride(), )*)
            }

            fn as_ptr(&self) -> Self::Ptr {
                let ($(ref $p,)*) = *self;
                ($($p.as_ptr(), )*)
            }
            unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
                let ($(ref $q ,)*) = *self;
                let ($($p,)*) = ptr;
                ($($q.as_ref($p),)*)
            }

            unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
                let ($(ref $p,)*) = *self;
                ($($p.uget_ptr(i), )*)
            }

            fn split_at(self, axis: Axis, index: Ix) -> (Self, Self) {
                let ($($p,)*) = self;
                let ($($p,)*) = (
                    $($p.split_at(axis, index), )*
                );
                (
                    ($($p.0,)*),
                    ($($p.1,)*)
                )
            }
        }
        )+
    }
}

zipt_impl!{
    [A ][ a],
    [A B][ a b],
    [A B C][ a b c],
    [A B C D][ a b c d],
    [A B C D E][ a b c d e],
    [A B C D E F][ a b c d e f],
}

macro_rules! map_impl {
    ($([$notlast:ident $($p:ident)*],)+) => {
        $(
        #[allow(non_snake_case)]
        impl<D: Dimension, $($p: NdProducer<Dim=D>),*> Zip<($($p,)*), D> {
            /// Apply a function to all elements of the input arrays,
            /// visiting elements in lock step.
            pub fn apply<F>(mut self, mut function: F)
                where F: FnMut($($p::Item),*)
            {
                self.apply_core((), move |(), args| {
                    let ($($p,)*) = args;
                    FoldWhile::Continue(function($($p),*))
                });
            }

            /// Apply a fold function to all elements of the input arrays,
            /// visiting elements in lock step.
            ///
            /// The fold continues while the return value is a
            /// `FoldWhile::Continue`.
            pub fn fold_while<F, Acc>(mut self, acc: Acc, mut function: F)
                -> FoldWhile<Acc>
                where F: FnMut(Acc, $($p::Item),*) -> FoldWhile<Acc>
            {
                self.apply_core(acc, move |acc, args| {
                    let ($($p,)*) = args;
                    function(acc, $($p),*)
                })
            }

            expand_if!(@bool [$notlast]

            /// Include the producer `p` in the Zip.
            ///
            /// ***Panics*** if `p`’s shape doesn’t match the Zip’s exactly.
            pub fn and<P>(self, p: P) -> Zip<($($p,)* P::Output, ), D>
                where P: IntoNdProducer<Dim=D>,
            {
                let array = p.into_producer();
                self.check(&array);
                let part_layout = array.layout();
                let ($($p,)*) = self.parts;
                Zip {
                    parts: ($($p,)* array, ),
                    layout: self.layout.and(part_layout),
                    dimension: self.dimension,
                }
            }

            /// Include the producer `p` in the Zip, broadcasting if needed.
            ///
            /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
            ///
            /// ***Panics*** if broadcasting isn’t possible.
            pub fn and_broadcast<'a, P, D2, Elem>(self, p: P)
                -> Zip<($($p,)* ArrayView<'a, Elem, D>, ), D>
                where P: IntoNdProducer<Dim=D2, Output=ArrayView<'a, Elem, D2>, Item=&'a Elem>,
                      D2: Dimension,
            {
                let array = p.into_producer().broadcast_unwrap(self.dimension.clone());
                let part_layout = array.layout();
                let ($($p,)*) = self.parts;
                Zip {
                    parts: ($($p,)* array, ),
                    layout: self.layout.and(part_layout),
                    dimension: self.dimension,
                }
            }
            );

            /// Split the `Zip` evenly in two.
            ///
            /// It will be split in the way that best preserves element locality.
            pub fn split(self) -> (Self, Self) {
                debug_assert_ne!(self.size(), 0, "Attempt to split empty zip");
                debug_assert_ne!(self.size(), 1, "Attempt to split zip with 1 elem");
                // Always split in a way that preserves layout (if any)
                let axis = self.max_stride_axis();
                let index = self.len_of(axis) / 2;
                let (p1, p2) = self.parts.split_at(axis, index);
                let (d1, d2) = self.dimension.split_at(axis, index);
                (Zip {
                    dimension: d1,
                    layout: self.layout,
                    parts: p1,
                },
                Zip {
                    dimension: d2,
                    layout: self.layout,
                    parts: p2,
                })
            }
        }
        )+
    }
}

map_impl!{
    [true P1],
    [true P1 P2],
    [true P1 P2 P3],
    [true P1 P2 P3 P4],
    [true P1 P2 P3 P4 P5],
    [false P1 P2 P3 P4 P5 P6],
}

/// Value controlling the execution of `.fold_while` on `Zip`.
pub enum FoldWhile<T> {
    /// Continue folding with this value
    Continue(T),
    /// Fold is complete and will return this value
    Done(T),
}

impl<T> FoldWhile<T> {
    /// Return the inner value
    pub fn into_inner(self) -> T {
        match self {
            FoldWhile::Continue(x) | FoldWhile::Done(x) => x
        }
    }

    /// Return true if it is `Done`, false if `Continue`
    pub fn is_done(&self) -> bool {
        match *self {
            FoldWhile::Continue(_) => false,
            FoldWhile::Done(_) => true,
        }
    }
}
