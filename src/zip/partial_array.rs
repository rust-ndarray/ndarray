// Copyright 2020 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::imp_prelude::*;
use crate::{
    AssignElem,
    Layout,
    NdProducer,
    Zip,
    FoldWhile,
};

use std::cell::Cell;
use std::mem;
use std::mem::MaybeUninit;
use std::ptr;

/// An assignable element reference that increments a counter when assigned
pub(crate) struct ProxyElem<'a, 'b, A> {
    item: &'a mut MaybeUninit<A>,
    filled: &'b Cell<usize>
}

impl<'a, 'b, A> AssignElem<A> for ProxyElem<'a, 'b, A> {
    fn assign_elem(self, item: A) {
        self.filled.set(self.filled.get() + 1);
        *self.item = MaybeUninit::new(item);
    }
}

/// Handles progress of assigning to a part of an array, for elements that need
/// to be dropped on unwinding. See Self::scope.
pub(crate) struct PartialArray<'a, 'b, A, D>
    where D: Dimension
{
    data: ArrayViewMut<'a, MaybeUninit<A>, D>,
    filled: &'b Cell<usize>,
}

impl<'a, 'b, A, D> PartialArray<'a, 'b, A, D>
    where D: Dimension
{
    /// Create a temporary PartialArray that wraps the array view `data`;
    /// if the end of the scope is reached, the partial array is marked complete;
    /// if execution unwinds at any time before them, the elements written until then
    /// are dropped.
    ///
    /// Safety: the caller *must* ensure that elements will be written in `data`'s preferred order.
    /// PartialArray can not handle arbitrary writes, only in the memory order.
    pub(crate) unsafe fn scope(data: ArrayViewMut<'a, MaybeUninit<A>, D>,
                               scope_fn: impl FnOnce(&mut PartialArray<A, D>))
    {
        let filled = Cell::new(0);
        let mut partial = PartialArray::new(data, &filled);
        scope_fn(&mut partial);
        filled.set(0); // mark complete
    }

    unsafe fn new(data: ArrayViewMut<'a, MaybeUninit<A>, D>,
                  filled: &'b Cell<usize>) -> Self
    {
        debug_assert_eq!(filled.get(), 0);
        Self { data, filled }
    }
}

impl<'a, 'b, A, D> Drop for PartialArray<'a, 'b, A, D>
    where D: Dimension
{
    fn drop(&mut self) {
        if !mem::needs_drop::<A>() {
            return;
        }

        let mut count = self.filled.get();
        if count == 0 {
            return;
        }

        Zip::from(self).fold_while((), move |(), elt| {
            if count > 0 {
                count -= 1;
                unsafe {
                    ptr::drop_in_place::<A>(elt.item.as_mut_ptr());
                }
                FoldWhile::Continue(())
            } else {
                FoldWhile::Done(())
            }
        });
    }
}

impl<'a: 'c, 'b: 'c, 'c, A, D: Dimension> NdProducer for &'c mut PartialArray<'a, 'b, A, D> {
    // This just wraps ArrayViewMut as NdProducer and maps the item
    type Item = ProxyElem<'a, 'b, A>;
    type Dim = D;
    type Ptr = *mut MaybeUninit<A>;
    type Stride = isize;

    private_impl! {}
    fn raw_dim(&self) -> Self::Dim {
        self.data.raw_dim()
    }

    fn equal_dim(&self, dim: &Self::Dim) -> bool {
        self.data.equal_dim(dim)
    }

    fn as_ptr(&self) -> Self::Ptr {
        NdProducer::as_ptr(&self.data)
    }

    fn layout(&self) -> Layout {
        self.data.layout()
    }

    unsafe fn as_ref(&self, ptr: Self::Ptr) -> Self::Item {
        ProxyElem { filled: self.filled, item: &mut *ptr }
    }

    unsafe fn uget_ptr(&self, i: &Self::Dim) -> Self::Ptr {
        self.data.uget_ptr(i)
    }

    fn stride_of(&self, axis: Axis) -> Self::Stride {
        self.data.stride_of(axis)
    }

    #[inline(always)]
    fn contiguous_stride(&self) -> Self::Stride {
        self.data.contiguous_stride()
    }

    fn split_at(self, _axis: Axis, _index: usize) -> (Self, Self) {
        unimplemented!();
    }
}

