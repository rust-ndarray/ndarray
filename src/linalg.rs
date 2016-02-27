use libnum::{Zero, One};
use std::ops::{Add, Sub, Mul, Div};
use std::any::Any;

#[cfg(feature="rblas")]
use std::any::TypeId;

#[cfg(feature="rblas")]
use ShapeError;

#[cfg(feature="rblas")]
use blas::{AsBlas, BlasArrayView};

#[cfg(feature="rblas")]
use imp_prelude::*;

/// Trait union for scalars (array elements) that support linear algebra operations.
///
/// `Any` for type-based specialization, `Copy` so that they don't need move
/// semantics or destructors, and the rest are numerical traits.
pub trait LinalgScalar :
    Any +
    Copy +
    Zero + One +
    Add<Output=Self> +
    Sub<Output=Self> +
    Mul<Output=Self> +
    Div<Output=Self>
{ }

impl<T> LinalgScalar for T
    where T:
    Any +
    Copy +
    Zero + One +
    Add<Output=T> +
    Sub<Output=T> +
    Mul<Output=T> +
    Div<Output=T>
{ }

#[cfg(feature = "rblas")]
pub trait AsBlasAny<A, S, D> : AsBlas<A, S, D> {
    fn blas_view_as_type<T: Any>(&self) -> Result<BlasArrayView<T, D>, ShapeError>
        where S: Data;
}

#[cfg(feature = "rblas")]
/// ***Requires `features = "rblas"`***
impl<A, S, D> AsBlasAny<A, S, D> for ArrayBase<S, D>
    where S: Data<Elem=A>,
          D: Dimension,
          A: Any,
{
    fn blas_view_as_type<T: Any>(&self) -> Result<BlasArrayView<T, D>, ShapeError>
        where S: Data
    {
        if TypeId::of::<A>() == TypeId::of::<T>() {
            unsafe {
                let v = self.view();
                let u = ArrayView::new_(v.ptr as *const T, v.dim, v.strides);
                Priv(u).into_blas_view()
            }
        } else {
            Err(ShapeError::IncompatibleLayout)
        }
    }
}
