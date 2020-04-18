use std::mem::MaybeUninit;
use std::mem::transmute;

use crate::imp_prelude::*;
use crate::OwnedRepr;

/// Methods specific to `Array0`.
///
/// ***See also all methods for [`ArrayBase`]***
///
/// [`ArrayBase`]: struct.ArrayBase.html
impl<A> Array<A, Ix0> {
    /// Returns the single element in the array without cloning it.
    ///
    /// ```
    /// use ndarray::{arr0, Array0};
    ///
    /// // `Foo` doesn't implement `Clone`.
    /// #[derive(Debug, Eq, PartialEq)]
    /// struct Foo;
    ///
    /// let array: Array0<Foo> = arr0(Foo);
    /// let scalar: Foo = array.into_scalar();
    /// assert_eq!(scalar, Foo);
    /// ```
    pub fn into_scalar(self) -> A {
        let size = ::std::mem::size_of::<A>();
        if size == 0 {
            // Any index in the `Vec` is fine since all elements are identical.
            self.data.into_vec().remove(0)
        } else {
            // Find the index in the `Vec` corresponding to `self.ptr`.
            // (This is necessary because the element in the array might not be
            // the first element in the `Vec`, such as if the array was created
            // by `array![1, 2, 3, 4].slice_move(s![2])`.)
            let first = self.ptr.as_ptr() as usize;
            let base = self.data.as_ptr() as usize;
            let index = (first - base) / size;
            debug_assert_eq!((first - base) % size, 0);
            // Remove the element at the index and return it.
            self.data.into_vec().remove(index)
        }
    }
}

/// Methods specific to `Array`.
///
/// ***See also all methods for [`ArrayBase`]***
///
/// [`ArrayBase`]: struct.ArrayBase.html
impl<A, D> Array<A, D>
where
    D: Dimension,
{
    /// Return a vector of the elements in the array, in the way they are
    /// stored internally.
    ///
    /// If the array is in standard memory layout, the logical element order
    /// of the array (`.iter()` order) and of the returned vector will be the same.
    pub fn into_raw_vec(self) -> Vec<A> {
        self.data.into_vec()
    }
}

/// Methods specific to `Array` of `MaybeUninit`.
///
/// ***See also all methods for [`ArrayBase`]***
///
/// [`ArrayBase`]: struct.ArrayBase.html
impl<A, D> Array<MaybeUninit<A>, D>
where
    D: Dimension,
{
    /// Assert that the array's storage's elements are all fully initialized, and conver
    /// the array from element type `MaybeUninit<A>` to `A`.
    pub(crate) unsafe fn assume_init(self) -> Array<A, D> {
        // NOTE: Fully initialized includes elements not reachable in current slicing/view.
        //
        // Should this method be generalized to all array types?
        // (Will need a way to map the RawData<Elem=X> to RawData<Elem=Y> of same kind)

        let Array { data, ptr, dim, strides } = self;
        let data = transmute::<OwnedRepr<MaybeUninit<A>>, OwnedRepr<A>>(data);
        let ptr = ptr.cast::<A>();

        Array {
            data,
            ptr,
            dim,
            strides,
        }
    }
}
