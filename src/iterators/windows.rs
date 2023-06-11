use super::ElementsBase;
use crate::imp_prelude::*;
use crate::IntoDimension;
use crate::Layout;
use crate::NdProducer;

/// Window producer and iterable
///
/// See [`.windows()`](ArrayBase::windows) for more
/// information.
pub struct Windows<'a, A, D> {
    base: ArrayView<'a, A, D>,
    window: D,
    strides: D,
}

impl<'a, A, D: Dimension> Windows<'a, A, D> {
    pub(crate) fn new<E>(a: ArrayView<'a, A, D>, window_size: E) -> Self
    where
        E: IntoDimension<Dim = D>,
    {
        let window = window_size.into_dimension();
        let ndim = window.ndim();

        let mut unit_stride = D::zeros(ndim);
        unit_stride.slice_mut().fill(1);
        
        Windows::new_with_stride(a, window, unit_stride)
    }

    pub(crate) fn new_with_stride<E>(a: ArrayView<'a, A, D>, window_size: E, strides: E) -> Self
    where
        E: IntoDimension<Dim = D>,
    {
        let window = window_size.into_dimension();
        let strides_d = strides.into_dimension();
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
            a.ndim() == strides_d.ndim(),
            concat!(
                "Stride dimension {} does not match array dimension {} ",
                "(with array of shape {:?})"
            ),
            strides_d.ndim(),
            a.ndim(),
            a.shape()
        );
        let mut size = a.dim;
        for ((sz, &ws), &stride) in size
            .slice_mut()
            .iter_mut()
            .zip(window.slice())
            .zip(strides_d.slice())
        {
            assert_ne!(ws, 0, "window-size must not be zero!");
            assert_ne!(stride, 0, "stride cannot have a dimension as zero!");
            // cannot use std::cmp::max(0, ..) since arithmetic underflow panics
            *sz = if *sz < ws {
                0
            } else {
                ((*sz - (ws - 1) - 1) / stride) + 1
            };
        }
        let window_strides = a.strides.clone();

        let mut array_strides = a.strides.clone();
        for (arr_stride, ix_stride) in array_strides.slice_mut().iter_mut().zip(strides_d.slice()) {
            *arr_stride *= ix_stride;
        }

        unsafe {
            Windows {
                base: ArrayView::new(a.ptr, size, array_strides),
                window,
                strides: window_strides,
            }
        }
    }
}

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone ]
    Windows {
        base,
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
            iter: self.base.into_elements_base(),
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
    iter: ElementsBase<'a, A, D>,
    window: D,
    strides: D,
}

impl_iterator! {
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone]
    WindowsIter {
        iter,
        window,
        strides,
    }
    WindowsIter<'a, A, D> {
        type Item = ArrayView<'a, A, D>;

        fn item(&mut self, elt) {
            unsafe {
                ArrayView::new_(
                    elt,
                    self.window.clone(),
                    self.strides.clone())
            }
        }
    }
}
