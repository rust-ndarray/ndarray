use super::ElementsBase;
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

    pub(crate) fn new_with_stride<E>(a: ArrayView<'a, A, D>, window_size: E, axis_strides: E) -> Self
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

/// Window producer and iterable
///
/// See [`.axis_windows()`](ArrayBase::axis_windows) for more
/// information.
pub struct AxisWindows<'a, A, D>{
    base: ArrayView<'a, A, D>,
    window_size: usize,
    axis_idx: usize,
}

impl<'a, A, D: Dimension> AxisWindows<'a, A, D> {
    pub(crate) fn new(a: ArrayView<'a, A, D>, axis: Axis, window_size: usize) -> Self
    {
        let mut base = a;
        let len = base.raw_dim()[axis.index()];
        let indices = if len < window_size {
            Slice::new(0, Some(0), 1)
        } else {
            Slice::new(0, Some((len - window_size + 1) as isize), 1)
        };
        base.slice_axis_inplace(axis, indices);

        AxisWindows {
            base,
            window_size,
            axis_idx: axis.index(),
        }
    }

    fn window(&self) -> D{
        let mut window = self.base.raw_dim();
        window[self.axis_idx] = self.window_size;
        window
    }

    fn strides_(&self) -> D{
        let mut strides = D::zeros(self.base.ndim());
        strides.slice_mut().fill(1);
        strides
    }
}


impl<'a, A, D: Dimension> NdProducer for AxisWindows<'a, A, D> {
    type Item = ArrayView<'a, A, D>;
    type Dim = Ix1;
    type Ptr = *mut A;
    type Stride = isize;

    fn raw_dim(&self) -> Ix1 {
        Ix1(self.base.raw_dim()[self.axis_idx])
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
        ArrayView::new_(ptr, self.window(),
        self.strides_())
    }

    unsafe fn uget_ptr(&self, i: &Self::Dim) -> *mut A {
        let mut d = D::zeros(self.base.ndim());
        d[self.axis_idx] = i[0];
        self.base.uget_ptr(&d)
    }

    fn stride_of(&self, axis: Axis) -> isize {
        assert_eq!(axis, Axis(0));
        self.base.stride_of(Axis(self.axis_idx))
    }

    fn split_at(self, axis: Axis, index: usize) -> (Self, Self) {
        assert_eq!(axis, Axis(0));
        let (a, b) = self.base.split_at(Axis(self.axis_idx), index);
        (AxisWindows {
            base: a,
            window_size: self.window_size,
            axis_idx: self.axis_idx,

        },
        AxisWindows {
            base: b,
            window_size: self.window_size,
            axis_idx: self.axis_idx,
        })
    }

    private_impl!{}
}

impl<'a, A, D> IntoIterator for AxisWindows<'a, A, D>
where
    D: Dimension,
    A: 'a,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = WindowsIter<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter {
        let window =  self.window();
        let strides = self.strides_();
        WindowsIter {
            iter: self.base.into_elements_base(),
            window,
            strides,
        }
    }
}
