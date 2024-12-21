//! Traits for accepting multiple types as arrays.

use crate::{
    aview0,
    aview1,
    aview_mut1,
    ArrayBase,
    ArrayRef,
    ArrayViewMut,
    CowArray,
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
/// ## More Details
/// The key idea of `ArrayLike` is to bridge the gap between native ndarray types
/// (e.g., [`Array2`](crate::Array2), [`ArrayView`](crate::ArrayView)) and other
/// data structures like scalars or slices. It enables treating all these types
/// as "array-like" objects with a common set of operations.
///
/// # Example
/// ```
/// use ndarray::{array, Array, Array2, ArrayLike, DimMax};
///
/// fn multiply<A, T, G>(left: &T, right: &G) -> Array<A, <T::Dim as DimMax<G::Dim>>::Output>
/// where
///     T: ArrayLike<Elem = A>,
///     G: ArrayLike<Elem = B>,
///     T::Dim: DimMax<G::Dim>,
///     A: Mul,
/// {
///      left.as_array() * right.as_array()
/// }
///
/// let rows = array![[1], [2]];
/// let cols = array![3, 4];
/// assert_eq!(multiply(rows, col), array![[3, 4], [6, 8]]);
/// ```
pub trait ArrayLike
{
    type Dim: Dimension;
    type Elem;

    fn as_array(&self) -> CowArray<'_, Self::Elem, Self::Dim>;

    fn dim(&self) -> Self::Dim;

    fn as_elem(&self) -> Option<&Self::Elem>;
}

pub trait ArrayLikeMut: ArrayLike
{
    fn as_array_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>;

    fn as_elem_mut(&mut self) -> Option<&mut Self::Elem>;
}

impl<A> ArrayLike for A
where A: ScalarOperand
{
    type Dim = Ix0;
    type Elem = A;

    fn as_array(&self) -> CowArray<'_, Self::Elem, Self::Dim>
    where Self::Elem: Clone
    {
        aview0(self).into()
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
    fn as_array_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
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

    fn as_array(&self) -> CowArray<'_, Self::Elem, Self::Dim>
    {
        self.view().into()
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
    fn as_array_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
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

impl<S, D> ArrayLike for ArrayBase<S, D>
where
    S: Data,
    D: Dimension,
{
    type Dim = D;
    type Elem = S::Elem;

    fn as_array(&self) -> CowArray<'_, Self::Elem, Self::Dim>
    {
        self.into()
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
    fn as_array_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
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

impl<A> ArrayLike for [A]
{
    type Dim = Ix1;

    type Elem = A;

    fn as_array(&self) -> CowArray<'_, Self::Elem, Self::Dim>
    {
        aview1(self).into()
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
    fn as_array_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
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

    fn as_array(&self) -> CowArray<'_, Self::Elem, Self::Dim>
    {
        (&**self).as_array()
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
    fn as_array_mut(&mut self) -> ArrayViewMut<'_, Self::Elem, Self::Dim>
    {
        (&mut **self).as_array_mut()
    }

    fn as_elem_mut(&mut self) -> Option<&mut Self::Elem>
    {
        (&mut **self).as_elem_mut()
    }
}

#[cfg(test)]
mod tests
{

    use core::ops::Mul;

    use crate::{array, Array, ArrayLike, DimMax};

    fn multiply<T, G>(left: &T, right: &G) -> Array<T::Elem, <T::Dim as DimMax<G::Dim>>::Output>
    where
        T: ArrayLike,
        G: ArrayLike<Elem = T::Elem>,
        // Bounds to enable multiplication
        T::Elem: Clone + Mul<T::Elem, Output = T::Elem>,
        T::Dim: DimMax<G::Dim>,
    {
        let left = &*left.as_array();
        let right = &*right.as_array();
        left * right
    }

    #[test]
    fn test_multiply()
    {
        let left = array![1, 2];
        let right = array![3, 4];
        assert_eq!(multiply(&left, &right), array![3, 8]);
        assert_eq!(multiply(&left, &3), array![3, 6]);
    }
}
