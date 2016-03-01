
//! Constructor methods for ndarray
//!
use libnum;

use imp_prelude::*;
use dimension;
use linspace;
use error::{self, ShapeError};

/// Constructor methods for one-dimensional arrays.
impl<S> ArrayBase<S, Ix>
    where S: DataOwned
{
    /// Create a one-dimensional array from a vector (no allocation needed).
    pub fn from_vec(v: Vec<S::Elem>) -> ArrayBase<S, Ix> {
        unsafe { Self::from_vec_dim_unchecked(v.len() as Ix, v) }
    }

    /// Create a one-dimensional array from an iterable.
    pub fn from_iter<I: IntoIterator<Item=S::Elem>>(iterable: I) -> ArrayBase<S, Ix> {
        Self::from_vec(iterable.into_iter().collect())
    }

    /// Create a one-dimensional array from inclusive interval
    /// `[start, end]` with `n` elements. `F` must be a floating point type.
    pub fn linspace<F>(start: F, end: F, n: usize) -> ArrayBase<S, Ix>
        where S: Data<Elem=F>,
              F: libnum::Float,
    {
        Self::from_iter(linspace::linspace(start, end, n))
    }
}

/// Constructor methods for two-dimensional arrays.
impl<S, A> ArrayBase<S, (Ix, Ix)>
    where S: DataOwned<Elem=A>,
{
    /// Create an identity matrix of size `n` (square 2D array).
    ///
    /// **Panics** if `n * n` would overflow usize.
    pub fn eye(n: Ix) -> ArrayBase<S, (Ix, Ix)>
        where S: DataMut,
              A: Clone + libnum::Zero + libnum::One,
    {
        let mut eye = Self::zeros((n, n));
        for a_ii in eye.diag_mut() {
            *a_ii = A::one();
        }
        eye
    }
}

/// Constructor methods for arrays.
impl<S, A, D> ArrayBase<S, D>
    where S: DataOwned<Elem=A>,
          D: Dimension,
{
    /// Create an array with copies of `elem`, dimension `dim`.
    ///
    /// **Panics** if the number of elements in `dim` would overflow usize.
    ///
    /// ```
    /// use ndarray::RcArray;
    /// use ndarray::arr3;
    ///
    /// let a = RcArray::from_elem((2, 2, 2), 1.);
    ///
    /// assert!(
    ///     a == arr3(&[[[1., 1.],
    ///                  [1., 1.]],
    ///                 [[1., 1.],
    ///                  [1., 1.]]])
    /// );
    /// ```
    pub fn from_elem(dim: D, elem: A) -> ArrayBase<S, D>
        where A: Clone
    {
        // Note: We don't need to check the case of a size between
        // isize::MAX -> usize::MAX; in this case, the vec constructor itself
        // panics.
        let size = dim.size_checked().expect("Shape too large: overflow in size");
        let v = vec![elem; size];
        unsafe { Self::from_vec_dim_unchecked(dim, v) }
    }

    /// Create an array with copies of `elem`, dimension `dim` and fortran
    /// memory order.
    ///
    /// **Panics** if the number of elements would overflow usize.
    ///
    /// ```
    /// use ndarray::RcArray;
    /// use ndarray::arr3;
    ///
    /// let a = RcArray::from_elem_f((2, 2, 2), 1.);
    ///
    /// assert!(
    ///     a == arr3(&[[[1., 1.],
    ///                  [1., 1.]],
    ///                 [[1., 1.],
    ///                  [1., 1.]]])
    /// );
    /// assert!(a.strides() == &[1, 2, 4]);
    /// ```
    pub fn from_elem_f(dim: D, elem: A) -> ArrayBase<S, D>
        where A: Clone
    {
        let size = dim.size_checked().expect("Shape too large: overflow in size");
        let v = vec![elem; size];
        unsafe { Self::from_vec_dim_unchecked_f(dim, v) }
    }

    /// Create an array with zeros, dimension `dim`.
    ///
    /// **Panics** if the number of elements in `dim` would overflow usize.
    pub fn zeros(dim: D) -> ArrayBase<S, D>
        where A: Clone + libnum::Zero
    {
        Self::from_elem(dim, libnum::zero())
    }

    /// Create an array with zeros, dimension `dim` and fortran memory order.
    ///
    /// **Panics** if the number of elements in `dim` would overflow usize.
    pub fn zeros_f(dim: D) -> ArrayBase<S, D>
        where A: Clone + libnum::Zero
    {
        Self::from_elem_f(dim, libnum::zero())
    }

    /// Create an array with default values, dimension `dim`.
    ///
    /// **Panics** if the number of elements in `dim` would overflow usize.
    pub fn default(dim: D) -> ArrayBase<S, D>
        where A: Default
    {
        let v = (0..dim.size()).map(|_| A::default()).collect();
        unsafe { Self::from_vec_dim_unchecked(dim, v) }
    }

    /// Create an array from a vector (with no allocation needed).
    ///
    /// **Errors** if `dim` does not correspond to the number of elements
    /// in `v`.
    pub fn from_vec_dim(dim: D, v: Vec<A>) -> Result<ArrayBase<S, D>, ShapeError> {
        if dim.size_checked() != Some(v.len()) {
            return Err(error::incompatible_shapes(&v.len(), &dim));
        }
        unsafe { Ok(Self::from_vec_dim_unchecked(dim, v)) }
    }

    /// Create an array from a vector (with no allocation needed).
    ///
    /// Unsafe because dimension is unchecked, and must be correct.
    pub unsafe fn from_vec_dim_unchecked(dim: D, mut v: Vec<A>) -> ArrayBase<S, D> {
        debug_assert!(dim.size_checked() == Some(v.len()));
        ArrayBase {
            ptr: v.as_mut_ptr(),
            data: DataOwned::new(v),
            strides: dim.default_strides(),
            dim: dim,
        }
    }

    /// Create an array from a vector (with no allocation needed),
    /// using fortran memory order to interpret the data.
    ///
    /// Unsafe because dimension is unchecked, and must be correct.
    pub unsafe fn from_vec_dim_unchecked_f(dim: D, mut v: Vec<A>) -> ArrayBase<S, D> {
        debug_assert!(dim.size_checked() == Some(v.len()));
        ArrayBase {
            ptr: v.as_mut_ptr(),
            data: DataOwned::new(v),
            strides: dim.fortran_strides(),
            dim: dim,
        }
    }

    /// Create an array from a vector and interpret it according to the
    /// provided dimensions and strides. No allocation needed.
    ///
    /// Checks whether `dim` and `strides` are compatible with the vector's
    /// length, returning an `Err` if not compatible.
    ///
    /// **Errors** if strides and dimensions can point out of bounds of `v`.<br>
    /// **Errors** if strides allow multiple indices to point to the same element.
    pub fn from_vec_dim_stride(dim: D, strides: D, v: Vec<A>)
        -> Result<ArrayBase<S, D>, ShapeError>
    {
        dimension::can_index_slice(&v, &dim, &strides).map(|_| {
            unsafe {
                Self::from_vec_dim_stride_unchecked(dim, strides, v)
            }
        })
    }

    /// Create an array from a vector and interpret it according to the
    /// provided dimensions and strides. No allocation needed.
    ///
    /// Unsafe because dimension and strides are unchecked.
    pub unsafe fn from_vec_dim_stride_unchecked(dim: D, strides: D, mut v: Vec<A>)
        -> ArrayBase<S, D>
    {
        debug_assert!(dimension::can_index_slice(&v, &dim, &strides).is_ok());
        ArrayBase {
            ptr: v.as_mut_ptr(),
            data: DataOwned::new(v),
            strides: strides,
            dim: dim
        }
    }

}

