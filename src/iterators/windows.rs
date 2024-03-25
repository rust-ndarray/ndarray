use std::marker::PhantomData;

use super::Baseiter;
use crate::imp_prelude::*;
use crate::IntoDimension;
use crate::Layout;
use crate::NdProducer;
use crate::Slice;

/// Window producer and iterable
///
/// See [`.windows()`](ArrayBase::windows) for more
/// information.
pub struct Windows<'a, A, D> {
    base: RawArrayView<A, D>,
    life: PhantomData<&'a A>,
    window: D,
    strides: D,
}

impl<'a, A, D: Dimension> Windows<'a, A, D> {
    pub(crate) fn new<E>(a: ArrayView<'a, A, D>, window_size: E) -> Self
    where E: IntoDimension<Dim = D> {
        let window = window_size.into_dimension();
        let ndim = window.ndim();

        let mut unit_stride = D::zeros(ndim);
        unit_stride.slice_mut().fill(1);

        Windows::new_with_stride(a, window, unit_stride)
    }

    pub(crate) fn new_with_stride<E>(a: ArrayView<'a, A, D>, window_size: E, axis_strides: E) -> Self
    where E: IntoDimension<Dim = D> {
        let window = window_size.into_dimension();

        let strides = axis_strides.into_dimension();
        let window_strides = a.strides.clone();

        ndassert!(
            a.ndim() == window.ndim(),
            concat!(
                "Window dimension {} does not match array dimension {} ",
                "(with array of shape {:?})"
            ),
            window.ndim(),
            a.ndim(),
            a.shape()
        );

        ndassert!(
            a.ndim() == strides.ndim(),
            concat!(
                "Stride dimension {} does not match array dimension {} ",
                "(with array of shape {:?})"
            ),
            strides.ndim(),
            a.ndim(),
            a.shape()
        );

        let mut base = a;
        base.slice_each_axis_inplace(|ax_desc| {
            let len = ax_desc.len;
            let wsz = window[ax_desc.axis.index()];
            let stride = strides[ax_desc.axis.index()];

            if len < wsz {
                Slice::new(0, Some(0), 1)
            } else {
                Slice::new(0, Some((len - wsz + 1) as isize), stride as isize)
            }
        });

        Windows {
            base: base.into_raw_view(),
            life: PhantomData,
            window,
            strides: window_strides,
        }
    }
}

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone ]
    Windows {
        base,
        life,
        window,
        strides,
    }
    Windows<'a, A, D> {
        type Item = ArrayView<'a, A, D>;
        type Dim = D;

        unsafe fn item(&self, ptr) {
            ArrayView::new_(ptr, self.window.clone(),
                            self.strides.clone())
        }
    }
}

impl<'a, A, D> IntoIterator for Windows<'a, A, D>
where
    D: Dimension,
    A: 'a,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = WindowsIter<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter {
        WindowsIter {
            iter: self.base.into_base_iter(),
            life: self.life,
            window: self.window,
            strides: self.strides,
        }
    }
}

/// Window iterator.
///
/// See [`.windows()`](ArrayBase::windows) for more
/// information.
pub struct WindowsIter<'a, A, D> {
    iter: Baseiter<A, D>,
    life: PhantomData<&'a A>,
    window: D,
    strides: D,
}

impl_iterator! {
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone]
    WindowsIter {
        iter,
        life,
        window,
        strides,
    }
    WindowsIter<'a, A, D> {
        type Item = ArrayView<'a, A, D>;

        fn item(&mut self, ptr) {
            unsafe {
                ArrayView::new_(
                    ptr,
                    self.window.clone(),
                    self.strides.clone())
            }
        }
    }
}

send_sync_read_only!(Windows);
send_sync_read_only!(WindowsIter);
