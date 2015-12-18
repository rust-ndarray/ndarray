//! Experimental BLAS (Basic Linear Algebra Subprograms) integration
//!
//! ***Requires `features = "rblas"`***
//!
//! Depends on crate [`rblas`], ([docs]).
//!
//! [`rblas`]: https://crates.io/crates/rblas/
//! [docs]: http://mikkyang.github.io/rust-blas/doc/rblas/
//!
//! ```
//! extern crate rblas;
//! extern crate ndarray;
//!
//! use rblas::Gemv;
//! use rblas::attribute::Transpose;
//!
//! use ndarray::{arr1, arr2, Array};
//! use ndarray::blas_support::AsBlas;
//!
//! fn main() {
//!     let mut a = arr2(&[[1., 2., 3.],
//!                        [4., 5., 6.],
//!                        [7., 8., 9.]]);
//!     let mut y = arr1(&[0., 0., 0.]);
//!
//!     // gemv is y = α a x + β y
//!
//!     let x = [1., 0., 1.];
//!     Gemv::gemv(Transpose::NoTrans, &1., &a.blas(), &x[..], &1.,
//!                &mut y.blas_overwrite());
//!     assert_eq!(y, arr1(&[4., 10., 16.]));
//! }
//!
//! ```
//!
//! Usage notes: Use checked or panicking methods to convert an array into
//! a view that implements rblas' `Vector` or `Matrix` traits.
//!
//! blas does not support strided arrays, so they can be copied into
//! c-contiguous layout automatically. blas has its own error reporting system
//! and will not panic on errors (that I know), instead output its own error
//! conditions. For example on dimension mismatch in a matrix multiplication.
//!
extern crate rblas;

use self::rblas::{
    Matrix,
    Vector,
};
use super::{
    ArrayBase,
    ArrayView,
    ArrayViewMut,
    Ix,
    ShapeError,
    Data,
    DataMut,
    DataOwned,
    Dimension,
};


/// ***Requires `features = "rblas"`***
pub struct BlasArrayView<'a, A: 'a, D>(ArrayView<'a, A, D>);
impl<'a, A, D: Copy> Copy for BlasArrayView<'a, A, D> { }
impl<'a, A, D: Clone> Clone for BlasArrayView<'a, A, D> {
    fn clone(&self) -> Self {
        BlasArrayView(self.0.clone())
    }
}

/// ***Requires `features = "rblas"`***
pub struct BlasArrayViewMut<'a, A: 'a, D>(ArrayViewMut<'a, A, D>);

impl<S, D> ArrayBase<S, D>
    where S: Data,
          D: Dimension
{
    fn size_check(&self) -> Result<(), ShapeError> {
        for &dim in self.shape() {
            if dim > (i32::max_value() as u32) {
                return Err(ShapeError::DimensionTooLarge(
                    self.shape().to_vec().into_boxed_slice()));
            }
        }
        Ok(())
    }

    fn contiguous_check(&self) -> Result<(), ShapeError> {
        // FIXME: handle transposed
        if self.is_standard_layout() {
            Ok(())
        } else {
            Err(ShapeError::IncompatibleLayout)
        }
    }
}

impl<'a, A, D> ArrayView<'a, A, D>
    where D: Dimension,
{
    /// ***Requires `features = "rblas"`***
    fn into_matrix(self) -> Result<BlasArrayView<'a, A, D>, ShapeError>
    {
        if self.dim.ndim() > 1 {
            try!(self.contiguous_check());
        }
        try!(self.size_check());
        Ok(BlasArrayView(self))
    }
}

impl<'a, A, D> ArrayViewMut<'a, A, D>
    where D: Dimension,
{
    fn into_matrix_mut(self) -> Result<BlasArrayViewMut<'a, A, D>, ShapeError>
    {
        if self.dim.ndim() > 1 {
            try!(self.contiguous_check());
        }
        try!(self.size_check());
        Ok(BlasArrayViewMut(self))
    }
}

/// Convert an array into a blas friendly wrapper.
///
/// ***Requires `features = "rblas"`***
pub trait AsBlas<A, S, D> {
    /// Return a contiguous array view implementing Vector (1D) or Matrix (2D)
    /// traits.
    ///
    /// Elements are copied if needed to produce a contiguous array.
    ///
    /// **Errors:** Produces an error if any dimension is larger than `i32::MAX`.
    fn blas_checked(&mut self) -> Result<BlasArrayView<A, D>, ShapeError>
        where S: DataOwned,
              A: Clone;

    /// Return a contiguous array view implementing Vector (1D) or Matrix (2D)
    /// traits.
    ///
    /// Elements are copied if needed to produce a contiguous array.
    ///
    /// **Errors:** Produces an error if any dimension is larger than `i32::MAX`.
    fn blas_mut_checked(&mut self) -> Result<BlasArrayViewMut<A, D>, ShapeError>
        where S: DataOwned + DataMut,
              A: Clone;

    /// Equivalent to `.blas_checked().unwrap()`
    ///
    /// **Panics** if there was a an error in `blas_checked`.
    fn blas(&mut self) -> BlasArrayView<A, D>
        where S: DataOwned<Elem=A>,
              A: Clone
    {
        self.blas_checked().unwrap()
    }

    /// Equivalent to `.blas_mut_checked().unwrap()`
    ///
    /// **Panics** if there was a an error in `blas_mut_checked`.
    fn blas_mut(&mut self) -> BlasArrayViewMut<A, D>
        where S: DataOwned<Elem=A> + DataMut,
              A: Clone
    {
        self.blas_mut_checked().unwrap()
    }

    /// Equivalent to `.blas_mut_checked().unwrap()`, except elements
    /// are not copied to make the array contiguous: instead just
    /// dimensions and strides are adjusted, and elements end up in
    /// arbitrary location. Useful if the content of the array doesn't matter.
    ///
    /// **Panics** if there was a an error in `blas_mut_checked`.
    fn blas_overwrite(&mut self) -> BlasArrayViewMut<A, D>
        where S: DataMut;
}

impl<A, S, D> AsBlas<A, S, D> for ArrayBase<S, D>
    where S: Data<Elem=A>,
          D: Dimension,
{
    fn blas_checked(&mut self) -> Result<BlasArrayView<A, D>, ShapeError>
        where S: DataOwned,
              A: Clone,
    {
        try!(self.size_check());
        if self.dim.ndim() > 1 {
            self.ensure_standard_layout();
        }
        self.view().into_matrix()
    }

    fn blas_mut_checked(&mut self) -> Result<BlasArrayViewMut<A, D>, ShapeError>
        where S: DataOwned + DataMut,
              A: Clone,
    {
        try!(self.size_check());
        if self.dim.ndim() > 1 {
            self.ensure_standard_layout();
        }
        self.view_mut().into_matrix_mut()
    }

    fn blas_overwrite(&mut self) -> BlasArrayViewMut<A, D>
        where S: DataMut,
    {
        self.size_check().unwrap();
        if self.dim.ndim() > 1 {
            self.force_standard_layout();
        }
        BlasArrayViewMut(self.view_mut())
    }
}

impl<'a, A> Vector<A> for BlasArrayView<'a, A, Ix> {
    fn len(&self) -> i32 {
        self.0.len() as i32
    }

    unsafe fn as_ptr(&self) -> *const A {
        self.0.ptr as *const _
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut A {
        panic!("BlasArrayView is not mutable");
    }

    // increment: stride
    fn inc(&self) -> i32 {
        self.0.strides as i32
    }
}

impl<'a, A> Vector<A> for BlasArrayViewMut<'a, A, Ix> {
    fn len(&self) -> i32 {
        self.0.len() as i32
    }

    unsafe fn as_ptr(&self) -> *const A {
        self.0.ptr
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut A {
        self.0.ptr
    }

    // increment: stride
    fn inc(&self) -> i32 {
        self.0.strides as i32
    }
}

impl<'a, A> Matrix<A> for BlasArrayView<'a, A, (Ix, Ix)> {
    fn rows(&self) -> i32 {
        self.0.dim().1 as i32
    }

    fn cols(&self) -> i32 {
        self.0.dim().0 as i32
    }

    unsafe fn as_ptr(&self) -> *const A {
        self.0.ptr as *const _
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut A {
        panic!("BlasArrayView is not mutable");
    }
}

impl<'a, A> Matrix<A> for BlasArrayViewMut<'a, A, (Ix, Ix)> {
    fn rows(&self) -> i32 {
        self.0.dim().1 as i32
    }

    fn cols(&self) -> i32 {
        self.0.dim().0 as i32
    }

    unsafe fn as_ptr(&self) -> *const A {
        self.0.ptr as *const _
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut A {
        self.0.ptr
    }
}
