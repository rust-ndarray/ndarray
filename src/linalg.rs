use libnum::{Zero, One, Float};
use std::any::Any;
use std::fmt;
use std::ops::{Add, Sub, Mul, Div};
#[cfg(feature="assign_ops")]
use std::ops::{
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
};
use ScalarOperand;

#[cfg(feature="rblas")]
use std::any::TypeId;

#[cfg(feature="rblas")]
use ShapeError;

#[cfg(feature="rblas")]
use error::{from_kind, ErrorKind};

#[cfg(feature="rblas")]
use blas::{AsBlas, BlasArrayView};

#[cfg(feature="rblas")]
use imp_prelude::*;

/// Elements that support linear algebra operations.
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

/// Floating-point element types `f32` and `f64`.
///
/// Trait `NdFloat` is only implemented for `f32` and `f64` but encompasses as
/// much float-relevant ndarray functionality as possible, including the traits
/// needed for linear algebra (`Any`) and for *right hand side* scalar
/// operations (`ScalarOperand`).
#[cfg(not(feature="assign_ops"))]
pub trait NdFloat :
    Float +
    fmt::Display + fmt::Debug + fmt::LowerExp + fmt::UpperExp +
    ScalarOperand + LinalgScalar
{ }

/// Floating-point element types `f32` and `f64`.
///
/// Trait `NdFloat` is only implemented for `f32` and `f64` but encompasses as
/// much float-relevant ndarray functionality as possible, including the traits
/// needed for linear algebra (`Any`) and for *right hand side* scalar
/// operations (`ScalarOperand`).
///
/// This trait can only be implemented by `f32` and `f64`.
#[cfg(feature="assign_ops")]
pub trait NdFloat :
    Float +
    AddAssign + SubAssign + MulAssign + DivAssign + RemAssign +
    fmt::Display + fmt::Debug + fmt::LowerExp + fmt::UpperExp +
    ScalarOperand + LinalgScalar
{ }

impl NdFloat for f32 { }
impl NdFloat for f64 { }


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
            Err(from_kind(ErrorKind::IncompatibleLayout))
        }
    }
}
