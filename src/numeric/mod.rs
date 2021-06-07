use crate::imp_prelude::*;
use num_complex::Complex;
use rawpointer::PointerExt;
use std::mem;
use std::ptr::NonNull;

mod impl_numeric;

impl<T, S, D> ArrayBase<S, D>
where
    S: Data<Elem = Complex<T>>,
    D: Dimension,
{
    /// Returns views of the real and imaginary components of the elements.
    ///
    /// ```
    /// use ndarray::prelude::*;
    /// use num_complex::{Complex, Complex64};
    ///
    /// let arr = array![
    ///     [Complex64::new(1., 2.), Complex64::new(3., 4.)],
    ///     [Complex64::new(5., 6.), Complex64::new(7., 8.)],
    ///     [Complex64::new(9., 10.), Complex64::new(11., 12.)],
    /// ];
    /// let Complex { re, im } = arr.view_re_im();
    /// assert_eq!(re, array![[1., 3.], [5., 7.], [9., 11.]]);
    /// assert_eq!(im, array![[2., 4.], [6., 8.], [10., 12.]]);
    /// ```
    pub fn view_re_im(&self) -> Complex<ArrayView<'_, T, D>> {
        debug_assert!(self.pointer_is_inbounds());

        let dim = self.dim.clone();

        // Double the strides. In the zero-sized element case and for axes of
        // length <= 1, we leave the strides as-is to avoid possible overflow.
        let mut strides = self.strides.clone();
        if mem::size_of::<T>() != 0 {
            for ax in 0..strides.ndim() {
                if dim[ax] > 1 {
                    strides[ax] *= 2;
                }
            }
        }

        let ptr_re: NonNull<T> = self.ptr.cast();
        let ptr_im: NonNull<T> = if self.is_empty() {
            // In the empty case, we can just reuse the existing pointer since
            // it won't be dereferenced anyway. It is not safe to offset by one
            // since the allocation may be empty.
            ptr_re
        } else {
            // In the nonempty case, we can safely offset into the first
            // (complex) element.
            unsafe { ptr_re.add(1) }
        };

        // `Complex` is `repr(C)` with only fields `re: T` and `im: T`. So, the
        // real components of the elements start at the same pointer, and the
        // imaginary components start at the pointer offset by one, with
        // exactly double the strides. The new, doubled strides still meet the
        // overflow constraints:
        //
        // - For the zero-sized element case, the strides are unchanged in
        //   units of bytes and in units of the element type.
        //
        // - For the nonzero-sized element case:
        //
        //   - In units of bytes, the strides are unchanged.
        //
        //   - Since `Complex<T>` for nonzero `T` is always at least 2 bytes,
        //     and the original strides did not overflow in units of bytes, we
        //     know that the new doubled strides will not overflow in units of
        //     `T`.
        unsafe {
            Complex {
                re: ArrayView::new(ptr_re, dim.clone(), strides.clone()),
                im: ArrayView::new(ptr_im, dim, strides),
            }
        }
    }

    /// Returns mutable views of the real and imaginary components of the elements.
    ///
    /// ```
    /// use ndarray::prelude::*;
    /// use num_complex::{Complex, Complex64};
    ///
    /// let mut arr = array![
    ///     [Complex64::new(1., 2.), Complex64::new(3., 4.)],
    ///     [Complex64::new(5., 6.), Complex64::new(7., 8.)],
    ///     [Complex64::new(9., 10.), Complex64::new(11., 12.)],
    /// ];
    ///
    /// let Complex { mut re, mut im } = arr.view_mut_re_im();
    /// assert_eq!(re, array![[1., 3.], [5., 7.], [9., 11.]]);
    /// assert_eq!(im, array![[2., 4.], [6., 8.], [10., 12.]]);
    ///
    /// re[[0, 1]] = 13.;
    /// im[[2, 0]] = 14.;
    ///
    /// assert_eq!(arr[[0, 1]], Complex64::new(13., 4.));
    /// assert_eq!(arr[[2, 0]], Complex64::new(9., 14.));
    /// ```
    pub fn view_mut_re_im(&mut self) -> Complex<ArrayViewMut<'_, T, D>>
    where
        S: DataMut,
    {
        self.ensure_unique();

        let dim = self.dim.clone();

        // Double the strides. In the zero-sized element case and for axes of
        // length <= 1, we leave the strides as-is to avoid possible overflow.
        let mut strides = self.strides.clone();
        if mem::size_of::<T>() != 0 {
            for ax in 0..strides.ndim() {
                if dim[ax] > 1 {
                    strides[ax] *= 2;
                }
            }
        }

        let ptr_re: NonNull<T> = self.ptr.cast();
        let ptr_im: NonNull<T> = if self.is_empty() {
            // In the empty case, we can just reuse the existing pointer since
            // it won't be dereferenced anyway. It is not safe to offset by one
            // since the allocation may be empty.
            ptr_re
        } else {
            // In the nonempty case, we can safely offset into the first
            // (complex) element.
            unsafe { ptr_re.add(1) }
        };

        // `Complex` is `repr(C)` with only fields `re: T` and `im: T`. So, the
        // real components of the elements start at the same pointer, and the
        // imaginary components start at the pointer offset by one, with
        // exactly double the strides. The new, doubled strides still meet the
        // overflow constraints:
        //
        // - For the zero-sized element case, the strides are unchanged in
        //   units of bytes and in units of the element type.
        //
        // - For the nonzero-sized element case:
        //
        //   - In units of bytes, the strides are unchanged.
        //
        //   - Since `Complex<T>` for nonzero `T` is always at least 2 bytes,
        //     and the original strides did not overflow in units of bytes, we
        //     know that the new doubled strides will not overflow in units of
        //     `T`.
        unsafe {
            Complex {
                re: ArrayViewMut::new(ptr_re, dim.clone(), strides.clone()),
                im: ArrayViewMut::new(ptr_im, dim, strides),
            }
        }
    }
}
