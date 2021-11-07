// Copyright 2021 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::imp_prelude::*;

use crate::data_traits::RawDataSubst;
use crate::math_cell::MathCell;

use std::cell::Cell;
use std::mem;
use std::num::Wrapping;
use std::num;
use std::ptr::NonNull;

use num_complex::Complex;


pub unsafe trait EqualRepresentation<To> { }

unsafe impl<T> EqualRepresentation<T> for T { }

macro_rules! unsafe_impl_trivial_transmute_t {
    ($($from:ty => $to:ty,)+) => {
        $(
unsafe impl<T> EqualRepresentation<$to> for $from { }
        )+

    }
}

macro_rules! unsafe_impl_trivial_transmute_bidir {
    ($(($from:ty) <=> $to:ty,)+) => {
        $(
unsafe impl EqualRepresentation<$to> for $from { }
unsafe impl EqualRepresentation<$from> for $to { }
        )+

    }
}

macro_rules! unsafe_impl_trivial_transmute {
    ($($from:ty => $to:ty,)+) => {
        $(
unsafe impl EqualRepresentation<$to> for $from { }
        )+

    }
}

unsafe_impl_trivial_transmute_t! {
    T => [T; 1],
    [T; 1] => T,
    // transmute from cell to T, but not the other way around
    Cell<T> => T,
    MathCell<T> => T,
    *mut T => *const T,
    *const T => *mut T,
    NonNull<T> => *mut T,
    NonNull<T> => *const T,
    Wrapping<T> => T,
    T => Wrapping<T>,
}

unsafe_impl_trivial_transmute_bidir! {
    (usize) <=> isize,
    (u128) <=> i128,
    (u64) <=> i64,
    (u32) <=> i32,
    (u16) <=> i16,
    (u8) <=> i8,
}

unsafe_impl_trivial_transmute! {
    // only from nonzero, not to
    num::NonZeroU8 => u8,
    num::NonZeroI8 => i8,
    num::NonZeroU16 => u16,
    num::NonZeroI16 => i16,
    num::NonZeroU32 => u32,
    num::NonZeroI32 => i32,
    num::NonZeroU64 => u64,
    num::NonZeroI64 => i64,
    num::NonZeroU128 => u128,
    num::NonZeroI128 => i128,
    num::NonZeroUsize => usize,
    num::NonZeroIsize => isize,
}


pub unsafe trait MultiElement : MultiElementExtended<<Self as MultiElement>::Elem> {
    type Elem;
}

pub unsafe trait MultiElementExtended<Elem> {
    const LEN: usize;
}

unsafe impl<A, const N: usize> MultiElement for [A; N] {
    type Elem = A;
}

unsafe impl<A, const N: usize> MultiElementExtended<A> for [A; N] {
    const LEN: usize = N;
}

unsafe impl<A> MultiElement for Complex<A> {
    type Elem = A;
}

unsafe impl<A> MultiElementExtended<A> for Complex<A> {
    const LEN: usize = 2;
}

macro_rules! multi_elem {
    ($($from:ty => $to:ty,)+) => {
        $(
unsafe impl MultiElementExtended<$to> for $from {
    const LEN: usize = mem::size_of::<$from>() / mem::size_of::<$to>();
}
        )+

    }
}

multi_elem! {
    usize => u8,
    isize => i8,
    u128 => i64,
    u128 => u64,
    u128 => i32,
    u128 => u32,
    u128 => i16,
    u128 => u16,
    u128 => i8,
    u128 => u8,
    u64 => i32,
    u64 => u32,
    u64 => i16,
    u64 => u16,
    u64 => i8,
    u64 => u8,
    u32 => i16,
    u32 => u16,
    u32 => i8,
    u32 => u8,
    u16 => i8,
    u16 => u8,
    i128 => i64,
    i128 => u64,
    i128 => i32,
    i128 => u32,
    i128 => i16,
    i128 => u16,
    i128 => i8,
    i128 => u8,
    i64 => i32,
    i64 => u32,
    i64 => i16,
    i64 => u16,
    i64 => i8,
    i64 => u8,
    i32 => i16,
    i32 => u16,
    i32 => i8,
    i32 => u8,
    i16 => i8,
    i16 => u8,
}

impl<'a, A, D> ArrayView<'a, A, D>
where
    D: Dimension,
    A: MultiElement,
{
    ///
    /// Note: expanding a zero-element array, `[A; 0]`, leads to a new axis of length zero,
    /// i.e. the result is an empty array view.
    ///
    /// **Panics** if the product of non-zero axis lengths overflows `isize`.
    pub fn expand(self, new_axis: Axis) -> ArrayView<'a, A::Elem, D::Larger> {
        self.expand_to::<A::Elem>(new_axis)
    }
}

impl<'a, A, D> ArrayView<'a, A, D>
where
    D: Dimension,
{
    ///
    /// Note: expanding a zero-element array, `[A; 0]`, leads to a new axis of length zero,
    /// i.e. the result is an empty array view.
    ///
    /// **Panics** if the product of non-zero axis lengths overflows `isize`.
    pub fn expand_to<T>(self, new_axis: Axis) -> ArrayView<'a, T, D::Larger>
        where A: MultiElementExtended<T>,
    {
        assert_eq!(mem::size_of::<A>(), mem::size_of::<T>() * A::LEN);

        let mut dim = self.dim.insert_axis(new_axis);
        let mut strides = self.strides.insert_axis(new_axis);

        // Double the strides. In the zero-sized element case and for axes of
        // length <= 1, we leave the strides as-is to avoid possible overflow.
        let len = A::LEN as isize;
        if mem::size_of::<T>() != 0 {
            for ax in 0..strides.ndim() {
                if Axis(ax) == new_axis {
                    continue;
                }
                if dim[ax] > 1 {
                    strides[ax] = ((strides[ax] as isize) * len) as usize;
                }
            }
        }
        dim[new_axis.index()] = A::LEN;

        // TODO nicer assertion
        crate::dimension::size_of_shape_checked(&dim).unwrap();

        // safe because
        // size still fits in isize;
        // new strides are adapted to new element type, inside the same allocation.
        unsafe {
            ArrayBase::from_data_ptr(self.data.data_subst(), self.ptr.cast())
                .with_strides_dim(strides, dim)
        }
    }
}
