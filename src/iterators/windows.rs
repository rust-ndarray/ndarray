
use imp_prelude::*;
use IntoDimension;

pub struct Windows<'a, A: 'a, D> {
    iter  : ::iter::Iter<'a, A, D>,
    window: D,
    stride: D,
}

pub fn windows<A, D, E>(a: ArrayView<A, D>, window_size: E) -> Windows<A, D>
	where D: Dimension,
          E: IntoDimension<Dim=D>,
{
    let window = window_size.into_dimension();
    ndassert!(a.ndim() == window.ndim(),
        concat!("Window dimension {} does not match array dimension {} ",
        "(with array of shape {:?})"),
        window.ndim(), a.ndim(), a.shape());
    let mut size = a.raw_dim();
    for (sz, &ws) in size.slice_mut().iter_mut().zip(window.slice())
    {
        if ws == 0 { panic!("window-size must not be zero!"); }
        // cannot use std::cmp::max(0, ..) since arithmetic underflow panics
        *sz = if *sz < ws { 0 } else { *sz - ws + 1 };
    }

    let mut strides = a.raw_dim();
    for (a, b) in strides.slice_mut().iter_mut().zip(a.strides()) {
        *a = *b as Ix;
    }

    let mult_strides = strides.clone();

    unsafe {
        Windows {
            iter  : ArrayView::from_shape_ptr(size.clone().strides(mult_strides), a.as_ptr()).into_iter(),
            window: window,
            stride: strides,
        }
    }
}

impl<'a, A, D> Iterator for Windows<'a, A, D>
    where D: Dimension,
{
    type Item = ArrayView<'a, A, D>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|elt| {
            unsafe {
                ArrayView::from_shape_ptr(self.window.clone().strides(self.stride.clone()), elt)
            }
        })
    }
}
