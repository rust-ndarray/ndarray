//! Traits for accepting multiple types as arrays.

use crate::{
    aview0,
    aview1,
    aview_mut1,
    ArrayBase,
    ArrayRef,
    ArrayView,
    ArrayViewMut,
    Data,
    DataMut,
    Dimension,
    Ix0,
    Ix1,
    ScalarOperand,
};

/// A trait for anything that can act like a multidimensional array.
///
/// This trait provides a unified interface for interacting with arrays, scalars, slices,
/// and other types that conceptually act like arrays. It's designed to make your functions
/// more flexible by letting them handle a wide range of types without extra boilerplate.
///
/// Like other parts of the `ndarray` crate, `ArrayLike` only works with scalars that implement
/// [`ScalarOperand`].
///
/// # Example
/// ```
/// use core::ops::Mul;
/// use ndarray::{array, Array, ArrayLike, DimMax};
///
/// fn multiply<T, G>(left: T, right: G) -> Array<T::Elem, <T::Dim as DimMax<G::Dim>>::Output>
/// where
///     T: ArrayLike,
///     G: ArrayLike<Elem = T::Elem>,
///     // Bounds to enable multiplication
///     T::Elem: Clone + Mul<T::Elem, Output = T::Elem>,
///     G::Elem: Clone,
///     T::Dim: DimMax<G::Dim>,
/// {
///     &left.view() * &right.view()
/// }
///
/// let left = array![1, 2];
/// let right = vec![3, 4];
/// // Array-vector multiplication
/// assert_eq!(multiply(&left, &right), array![3, 8]);
/// // Array-scalar multiplication
/// assert_eq!(multiply(&left, 3), array![3, 6]);
/// ```
///
/// # `ArrayLike` vs [`ArrayRef`]
/// Both `ArrayLike` and `ArrayRef` provide a kind of unifying abstraction for `ndarray`,
/// and both are useful for writing functions with `ndarray`. However, they should not be
/// used interchangeably. `ArrayLike` is ideal when you want to write generic functions
/// that work with anything that "looks" like a multidimensional array, even if it isn't
/// strictly an `ndarray` type. When you know that a given variable or argument will be an
/// `ndarray` type, use `ArrayRef` instead.
pub trait ArrayLike
{
    /// The dimensionality of the underlying array-like data structure.
    type Dim: Dimension;

    /// The element type of the underlying array-like data structure.
    type Elem;

    /// Get a read-only view of the underlying array-like data structure.
    ///
    /// This method should never re-allocate the underlying data.
    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>;

    /// Get the shape and strides of the underlying array-like data structure.
    ///
    /// # Example
    fn dim(&self) -> Self::Dim;

    /// If the underlying object only has one element, return it; otherwise return `None`.
    ///
    /// This method allows for optimizations when the `ArrayLike` value is actually a
    /// scalar, in which case one can avoid allocations and aid the compiler by not
    /// turning it into a full view.
    ///
    /// # Example
    /// ```rust
    /// use ndarray::{array, ArrayLike};
    ///
    /// let arr = array![1, 2, 3];
    /// let arr_single = array![1];
    /// let scalar = 1;
    ///
    /// matches!(arr.as_elem(), None);
    /// matches!(arr.as_elem(), Some(1));
    /// matches!(scalar.as_elem(), Some(1));
    /// ```
    ///
    /// # For Implementors:
    /// Array-like objects that can contain multiple elements are free to return `Some(_)`
    /// if and only if a runtime check determines there is only one element in the container.
    fn as_elem(&self) -> Option<&Self::Elem>;
}

/// A trait for mutable array-like objects.
///
/// This extends [`ArrayLike`] by providing mutable access to the underlying data.
/// Use it when you need to modify the contents of an array-like object.
///
/// ## More Details
/// `ArrayLikeMut` is designed for types that can provide mutable access to their elements.
/// For example, mutable slices and arrays implement this trait, but immutable views or
/// read-only data structures won't.
///
/// # Examples
/// ```
/// use core::ops::MulAssign;
/// use ndarray::{array, ArrayLike, ArrayLikeMut, DimMax};
///
/// fn multiply_assign<T, G>(left: &mut T, right: &G)
/// where
///     T: ArrayLikeMut,
///     G: ArrayLike<Elem = T::Elem>,
///     // Bounds to enable multiplication
///     T::Elem: Clone + MulAssign<T::Elem>,
///     G::Elem: Clone,
///     // Ensure that the broadcast is still assignable to the left side
///     T::Dim: DimMax<G::Dim, Output = T::Dim>,
/// {
///     *left.view_mut() *= &right.view();
/// }
///
/// let mut left = array![1, 2];
/// let right = array![3, 4];
///
/// multiply_assign(&mut left, &right);
/// assert_eq!(left, array![3, 8]);
///
/// multiply_assign(&mut left, &2);
/// assert_eq!(left, array![6, 16]);
/// ```
pub trait ArrayLikeMut: ArrayLike
{
    /// Get a mutable view of the underlying array-like data structure.
    ///
    /// This method should never re-allocate the underlying data.
    fn view_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>;

    /// If the underlying object only has one element, return a mutable reference; otherwise return `None`.
    ///
    /// This method allows for optimizations when the `ArrayLike` value is actually a
    /// scalar, in which case one can avoid allocations and aid the compiler by not
    /// turning it into a full view.
    ///
    /// # Example
    /// ```rust
    /// use ndarray::{array, ArrayLike, ArrayLikeMut};
    /// use num_traits::Zero;
    ///
    /// fn assign_sum<T, G>(mut left: T, right: G)
    /// where
    ///     T: ArrayLikeMut,
    ///     G: ArrayLike<Elem = T::Elem>,
    ///     // Bounds to enable sum
    ///     T::Elem: Zero + Clone,
    /// {
    ///     if let Some(e) = left.as_elem_mut() {
    ///         *e = right.view().sum();
    ///     }
    /// }
    ///
    ///
    /// let arr = array![1, 2, 3];
    /// let mut arr_single = array![1];
    /// assign_sum(&mut arr_single, arr);
    /// assert_eq!(arr_single[0], 6);
    /// ```
    ///
    /// # For Implementors:
    /// Array-like objects that can contain multiple elements are free to return `Some(_)`
    /// if and only if a runtime check determines there is only one element in the container.
    fn as_elem_mut(&mut self) -> Option<&mut Self::Elem>;
}

impl<A> ArrayLike for A
where A: ScalarOperand
{
    type Dim = Ix0;
    type Elem = A;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    where Self::Elem: Clone
    {
        aview0(self)
    }

    fn dim(&self) -> Self::Dim
    {
        Ix0()
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        Some(self)
    }
}

impl<A> ArrayLikeMut for A
where A: ScalarOperand
{
    fn view_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
    {
        // SAFETY: The pointer will be non-null since it's a reference,
        // and the view is tied to the lifetime of the mutable borrow
        unsafe { ArrayViewMut::from_shape_ptr((), self as *mut Self::Elem) }
    }

    fn as_elem_mut(&mut self) -> Option<&mut Self::Elem>
    {
        Some(self)
    }
}

impl<A, D> ArrayLike for ArrayRef<A, D>
where D: Dimension
{
    type Dim = D;
    type Elem = A;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    {
        self.view()
    }

    fn dim(&self) -> Self::Dim
    {
        self.raw_dim()
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        if self.dim.size() == 1 {
            self.first()
        } else {
            None
        }
    }
}

impl<A, D> ArrayLike for &ArrayRef<A, D>
where D: Dimension
{
    type Dim = D;
    type Elem = A;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    {
        (*self).view()
    }

    fn dim(&self) -> Self::Dim
    {
        self.raw_dim()
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        if self.dim.size() == 1 {
            self.first()
        } else {
            None
        }
    }
}

impl<A, D> ArrayLike for &mut ArrayRef<A, D>
where D: Dimension
{
    type Dim = D;
    type Elem = A;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    {
        (**self).view()
    }

    fn dim(&self) -> Self::Dim
    {
        self.raw_dim()
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        if self.dim.size() == 1 {
            self.first()
        } else {
            None
        }
    }
}

impl<A, D> ArrayLikeMut for ArrayRef<A, D>
where D: Dimension
{
    fn view_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
    {
        self.view_mut()
    }

    fn as_elem_mut(&mut self) -> Option<&mut Self::Elem>
    {
        if self.dim.size() == 1 {
            self.first_mut()
        } else {
            None
        }
    }
}

impl<A, D> ArrayLikeMut for &mut ArrayRef<A, D>
where D: Dimension
{
    fn view_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
    {
        ArrayRef::view_mut(self)
    }

    fn as_elem_mut(&mut self) -> Option<&mut Self::Elem>
    {
        if self.dim.size() == 1 {
            self.first_mut()
        } else {
            None
        }
    }
}

impl<S, D> ArrayLike for ArrayBase<S, D>
where
    S: Data,
    D: Dimension,
{
    type Dim = D;
    type Elem = S::Elem;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    {
        ArrayRef::view(self)
    }

    fn dim(&self) -> Self::Dim
    {
        self.raw_dim()
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        if self.dim.size() == 1 {
            self.first()
        } else {
            None
        }
    }
}

impl<S, D> ArrayLike for &ArrayBase<S, D>
where
    S: Data,
    D: Dimension,
{
    type Dim = D;
    type Elem = S::Elem;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    {
        ArrayRef::view(self)
    }

    fn dim(&self) -> Self::Dim
    {
        self.raw_dim()
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        if self.dim.size() == 1 {
            self.first()
        } else {
            None
        }
    }
}

impl<S, D> ArrayLike for &mut ArrayBase<S, D>
where
    S: Data,
    D: Dimension,
{
    type Dim = D;
    type Elem = S::Elem;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    {
        ArrayRef::view(self)
    }

    fn dim(&self) -> Self::Dim
    {
        self.raw_dim()
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        if self.dim.size() == 1 {
            self.first()
        } else {
            None
        }
    }
}

impl<S, D> ArrayLikeMut for ArrayBase<S, D>
where
    S: DataMut,
    D: Dimension,
{
    fn view_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
    {
        ArrayRef::view_mut(self)
    }

    fn as_elem_mut(&mut self) -> Option<&mut Self::Elem>
    {
        if self.dim.size() == 1 {
            self.first_mut()
        } else {
            None
        }
    }
}

impl<S, D> ArrayLikeMut for &mut ArrayBase<S, D>
where
    S: DataMut,
    D: Dimension,
{
    fn view_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
    {
        ArrayRef::view_mut(self)
    }

    fn as_elem_mut(&mut self) -> Option<&mut Self::Elem>
    {
        if self.dim.size() == 1 {
            self.first_mut()
        } else {
            None
        }
    }
}

impl<A> ArrayLike for [A]
{
    type Dim = Ix1;

    type Elem = A;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    {
        aview1(self)
    }

    fn dim(&self) -> Self::Dim
    {
        Ix1(self.len())
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        if self.len() == 1 {
            Some(&self[0])
        } else {
            None
        }
    }
}

impl<A> ArrayLike for &[A]
{
    type Dim = Ix1;

    type Elem = A;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    {
        aview1(self)
    }

    fn dim(&self) -> Self::Dim
    {
        Ix1(self.len())
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        if self.len() == 1 {
            Some(&self[0])
        } else {
            None
        }
    }
}

impl<A> ArrayLike for &mut [A]
{
    type Dim = Ix1;

    type Elem = A;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    {
        aview1(self)
    }

    fn dim(&self) -> Self::Dim
    {
        Ix1(self.len())
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        if self.len() == 1 {
            Some(&self[0])
        } else {
            None
        }
    }
}

impl<A> ArrayLikeMut for [A]
{
    fn view_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
    {
        aview_mut1(self)
    }

    fn as_elem_mut(&mut self) -> Option<&mut Self::Elem>
    {
        if self.len() == 1 {
            Some(&mut self[0])
        } else {
            None
        }
    }
}

impl<A> ArrayLikeMut for &mut [A]
{
    fn view_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
    {
        aview_mut1(self)
    }

    fn as_elem_mut(&mut self) -> Option<&mut Self::Elem>
    {
        if self.len() == 1 {
            Some(&mut self[0])
        } else {
            None
        }
    }
}

impl<A> ArrayLike for Vec<A>
{
    type Dim = Ix1;

    type Elem = A;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    {
        (&**self).view()
    }

    fn dim(&self) -> Self::Dim
    {
        (&**self).dim()
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        (&**self).as_elem()
    }
}

impl<A> ArrayLike for &Vec<A>
{
    type Dim = Ix1;

    type Elem = A;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    {
        (&**self).view()
    }

    fn dim(&self) -> Self::Dim
    {
        (&**self).dim()
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        (&**self).as_elem()
    }
}

impl<A> ArrayLike for &mut Vec<A>
{
    type Dim = Ix1;

    type Elem = A;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    {
        (&**self).view()
    }

    fn dim(&self) -> Self::Dim
    {
        (&**self).dim()
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        (&**self).as_elem()
    }
}

impl<A> ArrayLikeMut for Vec<A>
{
    fn view_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
    {
        (&mut **self).view_mut()
    }

    fn as_elem_mut(&mut self) -> Option<&mut Self::Elem>
    {
        (&mut **self).as_elem_mut()
    }
}

impl<A> ArrayLikeMut for &mut Vec<A>
{
    fn view_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
    {
        (&mut **self).view_mut()
    }

    fn as_elem_mut(&mut self) -> Option<&mut Self::Elem>
    {
        (&mut **self).as_elem_mut()
    }
}

impl<A, const N: usize> ArrayLike for [A; N]
{
    type Dim = Ix1;
    type Elem = A;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    {
        ArrayView::from(self)
    }

    fn dim(&self) -> Self::Dim
    {
        Ix1(N)
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        if N == 1 {
            Some(&self[0])
        } else {
            None
        }
    }
}

impl<A, const N: usize> ArrayLike for &[A; N]
{
    type Dim = Ix1;
    type Elem = A;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    {
        ArrayView::from(self)
    }

    fn dim(&self) -> Self::Dim
    {
        Ix1(N)
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        if N == 1 {
            Some(&self[0])
        } else {
            None
        }
    }
}

impl<A, const N: usize> ArrayLike for &mut [A; N]
{
    type Dim = Ix1;
    type Elem = A;

    fn view(&self) -> ArrayView<'_, Self::Elem, Self::Dim>
    {
        ArrayView::from(self)
    }

    fn dim(&self) -> Self::Dim
    {
        Ix1(N)
    }

    fn as_elem(&self) -> Option<&Self::Elem>
    {
        if N == 1 {
            Some(&self[0])
        } else {
            None
        }
    }
}

impl<A, const N: usize> ArrayLikeMut for [A; N]
{
    fn view_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
    {
        ArrayViewMut::from(self)
    }

    fn as_elem_mut(&mut self) -> Option<&mut Self::Elem>
    {
        if N == 1 {
            Some(&mut self[0])
        } else {
            None
        }
    }
}

impl<A, const N: usize> ArrayLikeMut for &mut [A; N]
{
    fn view_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
    {
        ArrayViewMut::from(self)
    }

    fn as_elem_mut(&mut self) -> Option<&mut Self::Elem>
    {
        if N == 1 {
            Some(&mut self[0])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests
{
    use core::ops::{Mul, MulAssign};

    use crate::{array, Array, ArrayLike, DimMax};

    use super::ArrayLikeMut;

    fn multiply<T, G>(left: T, right: G) -> Array<T::Elem, <T::Dim as DimMax<G::Dim>>::Output>
    where
        T: ArrayLike,
        G: ArrayLike<Elem = T::Elem>,
        // Bounds to enable multiplication
        T::Elem: Clone + Mul<T::Elem, Output = T::Elem>,
        G::Elem: Clone,
        T::Dim: DimMax<G::Dim>,
    {
        &left.view() * &right.view()
    }

    fn multiply_assign<T, G>(mut left: T, right: G)
    where
        T: ArrayLikeMut,
        G: ArrayLike<Elem = T::Elem>,
        // Bounds to enable multiplication
        T::Elem: Clone + MulAssign<T::Elem>,
        G::Elem: Clone,
        // Ensure that the broadcast is still assignable to the left side
        T::Dim: DimMax<G::Dim, Output = T::Dim>,
    {
        *left.view_mut() *= &right.view();
    }

    #[test]
    fn test_multiply()
    {
        let left = array![1, 2];
        let right = array![3, 4];
        assert_eq!(multiply(&left, &right), array![3, 8]);
        assert_eq!(multiply(&left, 3), array![3, 6]);
    }

    #[test]
    fn test_multiply_assign()
    {
        let mut left = array![1, 2];
        let right = array![3, 4];

        multiply_assign(&mut left, &right);
        assert_eq!(left, array![3, 8]);

        multiply_assign(&mut left, 2);
        assert_eq!(left, array![6, 16]);
    }
}
