// Copyright 2017 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod zipmacro;

use imp_prelude::*;
use IntoDimension;
use NdIndex;

/// Return if the expression is a break value.
macro_rules! fold_while {
    ($e:expr) => {
        match $e {
            FoldWhile::Continue(x) => x,
            x => return x,
        }
    }
}

/// Memory layout
#[derive(Copy, Clone, Debug)]
pub struct Layout(u32);

impl Layout {
    #[doc(hidden)]
    #[inline(always)]
    pub fn one_dimensional() -> Layout {
        Layout(CORDER | FORDER)
    }
    #[doc(hidden)]
    #[inline(always)]
    pub fn c() -> Layout {
        Layout(CORDER)
    }
    #[doc(hidden)]
    #[inline(always)]
    pub fn f() -> Layout {
        Layout(FORDER)
    }
    #[inline(always)]
    #[doc(hidden)]
    pub fn none() -> Layout {
        Layout(0)
    }
    #[inline(always)]
    fn is(self, flag: u32) -> bool {
        self.0 & flag != 0
    }
    #[inline(always)]
    fn and(self, flag: Layout) -> Layout {
        Layout(self.0 & flag.0)
    }

    #[inline(always)]
    fn flag(self) -> u32 {
        self.0
    }
}

const CORDER: u32 = 1 << 0;
const FORDER: u32 = 1 << 1;

//use ndarray::Axis;

trait LayoutImpl {
    fn layout_impl(&self) -> Layout;
}

/// Broadcast an array so that it acts like a larger size and/or shape array.
///
/// See [broadcasting][1] for more information.
///
/// [1]: struct.ArrayBase.html#broadcasting
trait Broadcast<E>
    where E: IntoDimension,
{
    type Output: Producer<Dim=E::Dim>;
    /// Broadcast the array to the new dimensions `shape`.
    ///
    /// ***Panics*** if broadcasting isn’t possible.
    fn broadcast_unwrap(self, shape: E) -> Self::Output;
    private_decl!{}
}

impl<S, D> LayoutImpl for ArrayBase<S, D>
    where S: Data,
          D: Dimension,
{
    fn layout_impl(&self) -> Layout {
        Layout(if self.is_standard_layout() {
            if self.ndim() <= 1 {
                FORDER | CORDER
            } else {
                CORDER
            }
        } else if self.as_slice_memory_order().is_some() {
            FORDER
        } else {
            0
        })
    }
}

impl<'a, L> LayoutImpl for &'a L where L: LayoutImpl
{
    fn layout_impl(&self) -> Layout {
        (**self).layout_impl()
    }
}

impl<'a, L> LayoutImpl for &'a mut L where L: LayoutImpl
{
    fn layout_impl(&self) -> Layout {
        (**self).layout_impl()
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

trait Splittable : Sized {
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
pub trait IntoProducer {
    type Dim: Dimension;
    type Output: Producer<Dim=Self::Dim>;
    fn into_producer(self) -> Self::Output;
}

impl<P> IntoProducer for P where P: Producer {
    type Dim = P::Dim;
    type Output = Self;
    fn into_producer(self) -> Self::Output { self }
}

/// A producer of an n-dimensional set of elements;
/// for example an array view, mutable array view or an iterator
/// that yields chunks.
///
/// Producers are used as a arguments to `Zip` and `azip!()`.
pub trait Producer {
    type Item;
    type Elem;
    /// Dimension type
    type Dim: Dimension;
    #[doc(hidden)]
    fn layout(&self) -> Layout;
    #[doc(hidden)]
    fn raw_dim(&self) -> Self::Dim;
    #[doc(hidden)]
    fn equal_dim(&self, dim: &Self::Dim) -> bool {
        self.raw_dim() == *dim
    }
    #[doc(hidden)]
    fn as_ptr(&self) -> *mut Self::Elem;
    #[doc(hidden)]
    unsafe fn as_ref(&self, *mut Self::Elem) -> Self::Item;
    #[doc(hidden)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut Self::Elem;
    #[doc(hidden)]
    fn stride_of(&self, axis: Axis) -> isize;
    #[doc(hidden)]
    #[inline(always)]
    fn contiguous_stride(&self) -> isize {
        1
    }
    #[doc(hidden)]
    fn split_at(self, axis: Axis, index: usize) -> (Self, Self) where Self: Sized;
    private_decl!{}
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
impl<'a, A: 'a, S, D> IntoProducer for &'a ArrayBase<S, D>
    where D: Dimension,
          S: Data<Elem=A>,
{
    type Dim = D;
    type Output = ArrayView<'a, A, D>;
    fn into_producer(self) -> Self::Output {
        self.view()
    }
}

/// A mutable array reference is an n-dimensional producer of mutable element
/// references (like ArrayViewMut).
impl<'a, A: 'a, S, D> IntoProducer for &'a mut ArrayBase<S, D>
    where D: Dimension,
          S: DataMut<Elem=A>,
{
    type Dim = D;
    type Output = ArrayViewMut<'a, A, D>;
    fn into_producer(self) -> Self::Output {
        self.view_mut()
    }
}

/// A slice is a one-dimensional producer
impl<'a, A: 'a> IntoProducer for &'a [A] {
    type Dim = Ix1;
    type Output = ArrayView1<'a, A>;
    fn into_producer(self) -> Self::Output {
        <_>::from(self)
    }
}

/// A mutable slice is a mutable one-dimensional producer
impl<'a, A: 'a> IntoProducer for &'a mut [A] {
    type Dim = Ix1;
    type Output = ArrayViewMut1<'a, A>;
    fn into_producer(self) -> Self::Output {
        <_>::from(self)
    }
}

/// A Vec is a one-dimensional producer
impl<'a, A: 'a> IntoProducer for &'a Vec<A> {
    type Dim = Ix1;
    type Output = ArrayView1<'a, A>;
    fn into_producer(self) -> Self::Output {
        <_>::from(self)
    }
}

/// A mutable Vec is a mutable one-dimensional producer
impl<'a, A: 'a> IntoProducer for &'a mut Vec<A> {
    type Dim = Ix1;
    type Output = ArrayViewMut1<'a, A>;
    fn into_producer(self) -> Self::Output {
        <_>::from(self)
    }
}

impl<'a, A, D> Producer for ArrayView<'a, A, D>
    where D: Dimension,
{
    type Item = &'a A;
    type Dim = D;
    type Elem = A;

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
        LayoutImpl::layout_impl(self)
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
    
    #[doc(hidden)]
    fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
        self.split_at(axis, index)
    }
}

impl<'a, A, D> Producer for ArrayViewMut<'a, A, D>
    where D: Dimension,
{
    type Item = &'a mut A;
    type Dim = D;
    type Elem = A;

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
        LayoutImpl::layout_impl(self)
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

    #[doc(hidden)]
    fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
        self.split_at(axis, index)
    }
}



/// N-ary lock step iteration or function application for arrays.
///
/// Zip allows matching several arrays to each other elementwise and applying
/// a function over all tuples of elements (one element from each input at
/// a time).
///
/// Arrays yield shared or mutable references according to how they are input
/// (as a shared reference to an array, mutable reference to an array, or
/// a read-only or read-write array view).
///
/// If all the input arrays are of the same memory order the zip performs
/// much better and the compiler can usually vectorize the loop.
///
/// The order elements are visited is not specified. The arrays don’t
/// have to have the same element type.
///
/// ```
/// use ndarray::Zip;
/// use ndarray::Array2;
///
/// type M = Array2<f64>;
///
/// let mut a = M::zeros((64, 64));
/// let b = M::zeros((64, 64));
/// let c = M::zeros((64, 64));
/// let d = M::zeros((64, 64));
///
/// Zip::from(&mut a).and(&b).and(&c).and(&d).apply(|w, &x, &y, &z| {
///     *w += x + y * z;
/// });
///
/// ```
#[derive(Debug, Clone)]
pub struct Zip<Parts, D> {
    parts: Parts,
    dimension: D,
    layout: Layout,
}

impl<P, D> Zip<(P, ), D>
    where D: Dimension,
          P: Producer<Dim=D>
{
    /// Create a new `Zip` from the input array `array`.
    ///
    /// The Zip will take the exact dimension of `array` and all inputs
    /// must have the same dimensions (or be broadcast to them).
    pub fn from<Part>(array: Part) -> Self
        where Part: IntoProducer<Dim=D, Output=P>
    {
        let array = array.into_producer();
        let dim = array.raw_dim();
        Zip {
            dimension: dim,
            layout: array.layout(),
            parts: (array, ),
        }
    }
}

impl<Parts, D> Zip<Parts, D>
    where D: Dimension,
{

    fn check<P>(&self, part: &P)
        where P: Producer<Dim=D>
    {
        debug_assert_eq!(&self.dimension, &part.raw_dim());
        assert!(part.equal_dim(&self.dimension));
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
            CORDER => self.dimension.slice().iter()
                          .position(|&len| len > 1).unwrap_or(0),
            FORDER => self.dimension.slice().iter()
                          .rposition(|&len| len > 1).unwrap_or(self.dimension.ndim() - 1),
            _ => 0,
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
                let ptr_i = ptrs.stride_offset(i, inner_strides);
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
                    let p = ptr.stride_offset(i, inner_strides);
                    acc = fold_while!(function(acc, self.parts.as_ref(p)));
                }
            }

            index_ = self.dimension.next_for(index);
        }
        self.dimension[unroll_axis] = inner_len;
        FoldWhile::Continue(acc)
    }
}

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


trait OffsetTuple {
    type Args;
    unsafe fn offset(self, off: isize) -> Self;
    unsafe fn stride_offset(self, index: usize, stride: Self::Args) -> Self;
}

impl<T> OffsetTuple for *mut T {
    type Args = isize;
    unsafe fn offset(self, off: isize) -> Self {
        self.offset(off)
    }

    unsafe fn stride_offset(self, index: usize, stride: isize) -> Self {
        self.offset(index as isize * stride)
    }
}

macro_rules! sub {
    ($i:ident [$($x:tt)*]) => { $($x)* };
}


macro_rules! offset_impl {
    ($([$($param:ident)*][ $($q:ident)*],)+) => {
        $(
        #[allow(non_snake_case)]
        impl<$($param: Offset),*> OffsetTuple for ($($param, )*) {
            type Args = ($(sub!($param [isize]),)*);
            unsafe fn offset(self, off: isize) -> Self {
                let ($($param, )*) = self;
                ($($param . offset(off),)*)
            }

            unsafe fn stride_offset(self, index: usize, stride: Self::Args) -> Self {
                let ($($param, )*) = self;
                let ($($q, )*) = stride;
                ($(Offset::stride_offset($param, index, $q),)*)
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
        impl<Dim: Dimension, $($p: Producer<Dim=Dim>),*> ZippableTuple for ($($p, )*) {
            type Item = ($($p::Item, )*);
            type Ptr = ($(*mut $p::Elem, )*);
            type Dim = Dim;
            type Stride = ($(sub!($p [isize]),)* );

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
    ($([$($p:ident)*],)+) => {
        $(
        #[allow(non_snake_case)]
        impl<D: Dimension, $($p: Producer<Dim=D>),*> Zip<($($p,)*), D> {
            /// Apply a function to all elements of the input arrays,
            /// visiting elements in lock step.
            pub fn apply<Func>(&mut self, mut function: Func)
                where Func: FnMut($($p::Item),*)
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
            pub fn fold_while<Func, Acc>(&mut self, acc: Acc, mut function: Func)
                -> FoldWhile<Acc>
                where Func: FnMut(Acc, $($p::Item),*) -> FoldWhile<Acc>
            {
                self.apply_core(acc, move |acc, args| {
                    let ($($p,)*) = args;
                    function(acc, $($p),*)
                })
            }

            /// Include the array `array` in the Zip.
            ///
            /// ***Panics*** if `array`’s shape doen't match the Zip’s exactly.
            pub fn and<Part>(self, array: Part) -> Zip<($($p,)* Part::Output, ), D>
                where Part: IntoProducer<Dim=D>,
            {
                let array = array.into_producer();
                self.check(&array);
                let part_layout = array.layout();
                let ($($p,)*) = self.parts;
                Zip {
                    parts: ($($p,)* array, ),
                    layout: self.layout.and(part_layout),
                    dimension: self.dimension,
                }
            }

            /// Include the array `array` in the Zip, broadcasting if needed.
            ///
            /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
            ///
            /// ***Panics*** if broadcasting isn’t possible.
            pub fn and_broadcast<'a, Part, D2, Elem>(self, array: Part)
                -> Zip<($($p,)* ArrayView<'a, Elem, D>, ), D>
                where Part: IntoProducer<Dim=D2, Output=ArrayView<'a, Elem, D2>>,
                      D2: Dimension,
            {
                let array = array.into_producer().broadcast_unwrap(self.dimension.clone());
                let part_layout = array.layout();
                let ($($p,)*) = self.parts;
                Zip {
                    parts: ($($p,)* array, ),
                    layout: self.layout.and(part_layout),
                    dimension: self.dimension,
                }
            }

            /// Split the `Zip` evenly in two
            pub fn split(self) -> (Self, Self)
            {
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
    [P1],
    [P1 P2],
    [P1 P2 P3],
    [P1 P2 P3 P4],
    [P1 P2 P3 P4 P5],
    [P1 P2 P3 P4 P5 P6],
}

pub enum FoldWhile<T> {
    Continue(T),
    Done(T),
}

impl<T> FoldWhile<T> {
    /// Return the inner value
    pub fn into_inner(self) -> T {
        match self {
            FoldWhile::Continue(x) | FoldWhile::Done(x) => x
        }
    }
}
