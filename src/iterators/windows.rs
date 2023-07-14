use super::ElementsBase;
use crate::imp_prelude::*;
use crate::IntoDimension;
use crate::Layout;
use crate::NdProducer;
use crate::Slice;

#[derive(Clone)]
pub struct GeneralWindow;
#[derive(Clone)]
pub struct AxisWindow{pub(crate) index: usize}

/// Window producer and iterable
///
/// See [`.windows()`](ArrayBase::windows) for more
/// information.
pub struct Windows<'a, A, D, V> {
    base: ArrayView<'a, A, D>,
    window: D,
    strides: D,
    variant: V,
}

impl<'a, A, D: Dimension, V> Windows<'a, A, D, V> {
    pub(crate) fn new<E>(a: ArrayView<'a, A, D>, window_size: E, variant: V) -> Self
    where
        E: IntoDimension<Dim = D>,
    {
        let window = window_size.into_dimension();
        let ndim = window.ndim();

        let mut unit_stride = D::zeros(ndim);
        unit_stride.slice_mut().fill(1);

        Windows::new_with_stride(a, window, unit_stride, variant)
    }

    pub(crate) fn new_with_stride<E>(
        a: ArrayView<'a, A, D>,
        window_size: E,
        axis_strides: E,
        variant: V,
    ) -> Self
    where
        E: IntoDimension<Dim = D>,
    {
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
            base,
            window,
            strides: window_strides,
            variant,
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
        variant,
    }
    Windows<'a, A, D, GeneralWindow> {
        type Item = ArrayView<'a, A, D>;
        type Dim = D;

        unsafe fn item(&self, ptr) {
            ArrayView::new_(ptr, self.window.clone(),
                            self.strides.clone())
        }
    }
}

impl<'a, A, D, V> IntoIterator for Windows<'a, A, D, V>
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

impl<'a, A, D: Dimension> NdProducer for Windows<'a, A, D, AxisWindow> {
    type Item = ArrayView<'a, A, D>;
    type Dim = Ix1;
    type Ptr = *mut A;
    type Stride = isize;

    fn raw_dim(&self) -> Ix1 {
        Ix1(self.base.raw_dim()[self.variant.index])
    }

    fn layout(&self) -> Layout {
        self.base.layout()
    }

    fn as_ptr(&self) -> *mut A {
        self.base.as_ptr() as *mut _
    }

    fn contiguous_stride(&self) -> isize {
        self.base.contiguous_stride()
    }

    unsafe fn as_ref(&self, ptr: *mut A) -> Self::Item {
        ArrayView::new_(ptr, self.window.clone(), self.strides.clone())
    }

    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A {
        let mut d = D::zeros(self.base.ndim());
        d[self.variant.index] = i[0];
        self.base.uget_ptr(&d)
    }

    fn stride_of(&self, axis: Axis) -> isize {
        assert_eq!(axis, Axis(0));
        self.base.stride_of(Axis(self.variant.index))
    }

    fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
        assert_eq!(axis, Axis(0));
        let (a, b) = self.base.split_at(Axis(self.variant.index), index);
        (
            Windows {
                base: a,
                window: self.window.clone(),
                strides: self.strides.clone(),
                variant: self.variant.clone(),
            },
            Windows {
                base: b,
                window: self.window,
                strides: self.strides,
                variant: self.variant,
            },
        )
    }

    private_impl! {}
}
