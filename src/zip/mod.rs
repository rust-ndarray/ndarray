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

/// Return if the expression is a break value.
macro_rules! try_control {
    ($e:expr) => {
        match $e {
            x => if x.should_break() {
                return x;
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Layout(u32);

impl Layout {
    fn is(self, flag: u32) -> bool {
        self.0 & flag != 0
    }
    fn and(self, flag: Layout) -> Layout {
        Layout(self.0 & flag.0)
    }
    #[cfg(experimental)]
    fn flag(self) -> u32 {
        self.0
    }
}

#[cfg(experimental)]
const BOTH: u32 = 0b11;
const CORDER: u32 = 1 << 0;
const FORDER: u32 = 1 << 1;
#[cfg(experimental)]
const NO_ORDER: u32 = 0;

//use ndarray::Axis;

trait LayoutImpl {
    fn layout(&self) -> Layout;
}

/// Broadcast an array so that it acts like a larger size and/or shape array.
///
/// See [broadcasting][1] for more information.
///
/// [1]: struct.ArrayBase.html#broadcasting
pub trait Broadcast<E>
    where E: IntoDimension,
{
    type Output: View<Dim=E::Dim>;
    /// Broadcast the array to the new dimensions `shape`.
    ///
    /// ***Panics*** if broadcasting isn’t possible.
    fn broadcast_unwrap(self, shape: E) -> Self::Output;
    private_decl!{}
}

impl<'a, S, D, E> Broadcast<E> for &'a ArrayBase<S, D>
    where S: 'a + Data,
          D: Dimension,
          E: IntoDimension,
{
    type Output = ArrayView<'a, S::Elem, E::Dim>;
    fn broadcast_unwrap(self, shape: E) -> Self::Output {
        (self).broadcast_unwrap(shape.into_dimension())
    }
    private_impl!{}
}

impl<S, D> LayoutImpl for ArrayBase<S, D>
    where S: Data,
          D: Dimension,
{
    fn layout(&self) -> Layout {
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
    fn layout(&self) -> Layout {
        (**self).layout()
    }
}

impl<'a, L> LayoutImpl for &'a mut L where L: LayoutImpl
{
    fn layout(&self) -> Layout {
        (**self).layout()
    }
}

impl<'a, A, D, E> Broadcast<E> for ArrayView<'a, A, D>
    where E: IntoDimension,
          D: 'a + Dimension,
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

#[cfg(experimental)]
trait Splittable : Sized {
    fn split_at(self, Axis, Ix) -> (Self, Self);
}

#[cfg(experimental)]
impl<I> Splittable for Dim<I>
    where Dim<I>: Dimension,
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

#[cfg(experimental)]
impl<'a, A, D> Splittable for ArrayView<'a, A, D>
    where D: Dimension,
{
    fn split_at(self, axis: Axis, index: Ix) -> (Self, Self) {
        self.split_at(axis, index)
    }
}

#[cfg(experimental)]
impl<'a, A, D> Splittable for ArrayViewMut<'a, A, D>
    where D: Dimension,
{
    fn split_at(self, axis: Axis, index: Ix) -> (Self, Self) {
        self.split_at(axis, index)
    }
}

/// An array view or a reference to an array
pub trait View {
    /// Element type
    type Elem;
    /// Native reference type (shared/mutable as appropriate)
    type Ref;
    /// Dimension type
    type Dim: Dimension;
    #[doc(hidden)]
    fn layout(&self) -> Layout;
    #[doc(hidden)]
    fn raw_dim(&self) -> &Self::Dim;
    #[doc(hidden)]
    fn as_ptr(&self) -> *mut Self::Elem;
    #[doc(hidden)]
    unsafe fn as_ref(*mut Self::Elem) -> Self::Ref;
    #[doc(hidden)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut Self::Elem;
    #[doc(hidden)]
    fn stride_of(&self, axis: Axis) -> isize;
    #[doc(hidden)]
    fn ensure_unique(&mut self) { }
    private_decl!{}
}

trait ZippableTuple {
    type Elem;
    type Ref;
    type Ptr: Offset<Args=Self::Stride> + Copy;
    type Dim: Dimension;
    type Stride: Copy;
    fn as_ptr(&self) -> Self::Ptr;
    unsafe fn as_ref(Self::Ptr) -> Self::Ref;
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr;
    fn stride_of(&self, index: usize) -> Self::Stride;
}

impl<'a, A: 'a, S, D> View for &'a ArrayBase<S, D>
    where D: Dimension,
          S: Data<Elem=A>,
{
    type Elem = A;
    type Ref = &'a A;
    type Dim = D;

    private_impl!{}
    #[doc(hidden)]
    fn raw_dim(&self) -> &Self::Dim {
        &self.dim
    }

    #[doc(hidden)]
    fn as_ptr(&self) -> *mut Self::Elem {
        (**self).as_ptr() as _
    }

    #[doc(hidden)]
    fn layout(&self) -> Layout {
        LayoutImpl::layout(*self)
    }

    #[doc(hidden)]
    unsafe fn as_ref(ptr: *mut Self::Elem) -> Self::Ref {
        &*ptr
    }

    #[doc(hidden)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut Self::Elem {
        (**self).uget(i.clone()) as *const _ as _
    }

    #[doc(hidden)]
    fn stride_of(&self, axis: Axis) -> isize {
        self.strides()[axis.index()]
    }
}

impl<'a, A: 'a, S, D> View for &'a mut ArrayBase<S, D>
    where D: Dimension,
          S: DataMut<Elem=A>,
{
    type Elem = A;
    type Ref = &'a mut A;
    type Dim = D;

    private_impl!{}
    #[doc(hidden)]
    fn raw_dim(&self) -> &Self::Dim {
        &self.dim
    }

    #[doc(hidden)]
    fn as_ptr(&self) -> *mut Self::Elem {
        (**self).as_ptr() as _
    }

    #[doc(hidden)]
    fn layout(&self) -> Layout {
        LayoutImpl::layout(*self)
    }

    #[doc(hidden)]
    unsafe fn as_ref(ptr: *mut Self::Elem) -> Self::Ref {
        &mut *ptr
    }

    #[doc(hidden)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut Self::Elem {
        (**self).uget(i.clone()) as *const _ as _
    }

    #[doc(hidden)]
    fn stride_of(&self, axis: Axis) -> isize {
        self.strides()[axis.index()]
    }

    #[doc(hidden)]
    fn ensure_unique(&mut self) {
        // calls ensure_unique for RcArray
        self.as_mut_ptr();
    }
}

impl<'a, A, D> View for ArrayView<'a, A, D>
    where D: Dimension,
{
    type Elem = A;
    type Ref = &'a A;
    type Dim = D;

    private_impl!{}
    #[doc(hidden)]
    fn raw_dim(&self) -> &Self::Dim {
        &self.dim
    }

    #[doc(hidden)]
    fn as_ptr(&self) -> *mut Self::Elem {
        self.as_ptr() as _
    }

    #[doc(hidden)]
    fn layout(&self) -> Layout {
        LayoutImpl::layout(self)
    }

    #[doc(hidden)]
    unsafe fn as_ref(ptr: *mut Self::Elem) -> Self::Ref {
        &*ptr
    }

    #[doc(hidden)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut Self::Elem {
        self.uget(i.clone()) as *const _ as _
    }

    #[doc(hidden)]
    fn stride_of(&self, axis: Axis) -> isize {
        self.strides()[axis.index()]
    }
}

impl<'a, A, D> View for ArrayViewMut<'a, A, D>
    where D: Dimension,
{
    type Elem = A;
    type Ref = &'a mut A;
    type Dim = D;

    private_impl!{}
    #[doc(hidden)]
    fn raw_dim(&self) -> &Self::Dim {
        &self.dim
    }

    #[doc(hidden)]
    fn as_ptr(&self) -> *mut Self::Elem {
        self.as_ptr() as _
    }

    #[doc(hidden)]
    fn layout(&self) -> Layout {
        LayoutImpl::layout(self)
    }

    #[doc(hidden)]
    unsafe fn as_ref(ptr: *mut Self::Elem) -> Self::Ref {
        &mut *ptr
    }

    #[doc(hidden)]
    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut Self::Elem {
        self.uget(i.clone()) as *const _ as _
    }

    #[doc(hidden)]
    fn stride_of(&self, axis: Axis) -> isize {
        self.strides()[axis.index()]
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
          P: View<Dim=D>
{
    /// Create a new `Zip` from the input array `array`.
    ///
    /// The Zip will take the exact dimension of `array` and all inputs
    /// must have the same dimensions (or be broadcast to them).
    pub fn from(array: P) -> Self
    {
        let dim = array.raw_dim().clone();
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

    fn check<P>(&self, part: &mut P)
        where P: View<Dim=D>
    {
        debug_assert_eq!(&self.dimension, part.raw_dim());
        assert!(self.dimension.equal(part.raw_dim()));
        part.ensure_unique();
    }

    fn prepare<P>(&self, part: P) -> P::Output
        where P: Broadcast<D>,
    {
        let ret = part.broadcast_unwrap(self.dimension.clone());
        ret
    }

    #[cfg(experimental)]
    fn dim(&self) -> D::Pattern {
        self.dimension.clone().into_pattern()
    }

    #[cfg(experimental)]
    fn raw_dim(&self) -> &D {
        &self.dimension
    }

    #[cfg(experimental)]
    /// Return the length of `axis`
    ///
    /// ***Panics*** if `axis` is out of bounds.
    fn len_of(&self, axis: Axis) -> usize {
        self.dimension[axis.index()]
    }

    #[cfg(experimental)]
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

    #[cfg(experimental)]
    fn split_at(self, axis: Axis, index: Ix) -> (Self, Self)
        where Parts: Splittable,
              D: Splittable,
    {
        let (p1, p2) = self.parts.split_at(axis, index);
        let (d1, d2) = self.dimension.split_at(axis, index);
        let mut dim_layout = NO_ORDER;
        let ndim = d1.ndim();
        if ndim <= 1 || index == 0 {
            dim_layout |= BOTH;
        } else {
            if axis == Axis(0) ||
                d1.slice()[..axis.index()].iter().all(|&l| l == 1)
            {
                dim_layout |= CORDER
            }
            if axis == Axis(ndim - 1) ||
                d1.slice()[axis.index() + 1..].iter().all(|&l| l == 1)
            {
                dim_layout |= FORDER
            }
        }
        (Zip {
            dimension: d1,
            layout: self.layout.and(Layout(dim_layout)),
            parts: p1,
        },
        Zip {
            dimension: d2,
            layout: self.layout.and(Layout(dim_layout)),
            parts: p2,
        })
    }
}

impl<P, D> Zip<P, D>
    where D: Dimension,
{
    fn apply_core<F, R>(&mut self, function: F) -> R
        where F: FnMut(P::Ref) -> R,
              R: ControlFlow,
              P: ZippableTuple<Dim=D>,
    {
        if self.layout.is(CORDER | FORDER) {
            self.apply_core_contiguous(function)
        } else {
            self.apply_core_strided(function)
        }
    }
    fn apply_core_contiguous<F, R>(&mut self, mut function: F) -> R
        where F: FnMut(P::Ref) -> R,
              R: ControlFlow,
              P: ZippableTuple<Dim=D>,
    {
        debug_assert!(self.layout.is(CORDER | FORDER));
        let size = self.dimension.size();
        let ptrs = self.parts.as_ptr();
        for i in 0..size {
            unsafe {
                let ptr_i = ptrs.offset(i as isize);
                try_control![function(P::as_ref(ptr_i))];
            }
        }
        R::continuing()
    }

    fn apply_core_strided<F, R>(&mut self, mut function: F) -> R
        where F: FnMut(P::Ref) -> R,
              R: ControlFlow,
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
                    let p = ptr.offset_stride(i, inner_strides);
                    try_control!(function(P::as_ref(p)));
                }
            }

            index_ = self.dimension.next_for(index);
        }
        self.dimension[unroll_axis] = inner_len;
        R::continuing()
    }
}

trait Offset {
    type Args;
    unsafe fn offset(self, off: isize) -> Self;
    unsafe fn offset_stride(self, index: usize, stride: Self::Args) -> Self;
}

impl<T> Offset for *mut T {
    type Args = isize;
    unsafe fn offset(self, off: isize) -> Self {
        self.offset(off)
    }

    unsafe fn offset_stride(self, index: usize, stride: isize) -> Self {
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
        impl<$($param),*> Offset for ($(*mut $param, )*) {
            type Args = ($(sub!($param [isize]),)*);
            unsafe fn offset(self, off: isize) -> Self {
                let ($($param, )*) = self;
                ($($param . offset(off),)*)
            }

            unsafe fn offset_stride(self, index: usize, stride: Self::Args) -> Self {
                let ($($param, )*) = self;
                let ($($q, )*) = stride;
                ($(Offset::offset_stride($param, index, $q),)*)
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
    ($([$($p:ident)*],)+) => {
        $(
        #[allow(non_snake_case)]
        impl<Dim: Dimension, $($p: View<Dim=Dim>),*> ZippableTuple for ($($p, )*) {
            type Elem = ($($p::Elem, )*);
            type Ref = ($($p::Ref, )*);
            type Ptr = ($(*mut $p::Elem, )*);
            type Dim = Dim;
            type Stride = ($(sub!($p [isize]),)* );

            fn stride_of(&self, index: usize) -> Self::Stride {
                let ($(ref $p,)*) = *self;
                ($($p.stride_of(Axis(index)), )*)
            }

            fn as_ptr(&self) -> Self::Ptr {
                let ($(ref $p,)*) = *self;
                ($($p.as_ptr(), )*)
            }
            unsafe fn as_ref(ptr: Self::Ptr) -> Self::Ref {
                let ($($p,)*) = ptr;
                ($($p::as_ref($p),)*)
            }

            unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
                let ($(ref $p,)*) = *self;
                ($($p.uget_ptr(i), )*)
            }
        }
        )+
    }
}

zipt_impl!{
    [A],
    [A B],
    [A B C],
    [A B C D],
    [A B C D E],
    [A B C D E F],
}

macro_rules! map_impl {
    ($([$($p:ident)*],)+) => {
        $(
        #[allow(non_snake_case)]
        impl<Dim: Dimension, $($p: View<Dim=Dim>),*> Zip<($($p,)*), Dim> {
            /// Apply a function to all elements of the input arrays,
            /// visiting elements in lock step.
            pub fn apply<Func>(&mut self, mut function: Func)
                where Func: FnMut($($p::Ref),*)
            {
                self.apply_core(move |args| {
                    let ($($p,)*) = args;
                    function($($p),*)
                })
            }

            /// Include the array `array` in the Zip.
            ///
            /// ***Panics*** if `array`’s shape doen't match the Zip’s exactly.
            pub fn and<Part>(self, mut array: Part) -> Zip<($($p,)* Part, ), Dim>
                where Part: View<Dim=Dim>,
            {
                self.check(&mut array);
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
            pub fn and_broadcast<Part>(self, array: Part) -> Zip<($($p,)* Part::Output, ), Dim>
                where Part: Broadcast<Dim>,
            {
                let array = self.prepare(array);
                let part_layout = array.layout();
                let ($($p,)*) = self.parts;
                Zip {
                    parts: ($($p,)* array, ),
                    layout: self.layout.and(part_layout),
                    dimension: self.dimension,
                }
            }
        }
        )+
    }
}

map_impl!{
    [A],
    [A B],
    [A B C],
    [A B C D],
    [A B C D E],
    [A B C D E F],
}

macro_rules! split_impl {
    ([]) => { };
    ([$($p:ident)+]) => {
        split_impl!{@recur [$($p)*]}
        #[allow(non_snake_case)]
        impl<$($p: Splittable),*> Splittable for ($($p,)*) {
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
    };
    (@recur [$p1:ident $($p:ident)*]) => {
        split_impl!([$($p)*]);
    };
}

#[cfg(experimental)]
split_impl!{
    [A B C D E F]
}

/// Control flow for callbacks.
///
/// **ControlFlow** allows breaking and returning using a callback, as an opt-in
/// feature. The default return value `()` means that the callback should
/// always continue.
///
/// Use the `Control` enum to have early break behavior.
trait ControlFlow {
    fn continuing() -> Self;
    fn should_break(&self) -> bool;
}

impl ControlFlow for () {
    fn continuing() { }
    #[inline]
    fn should_break(&self) -> bool { false }
}
