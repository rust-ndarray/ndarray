// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::any::Any;
use libnum::Complex;

/// Elements that can be used as direct operands in arithmetic with arrays.
///
/// For example, `f64` is a `ScalarOperand` which means that for an array `a`,
/// arithmetic like `a + 1.0`, and, `a * 2.`, and `a += 3.` are allowed.
///
/// In the description below, let `A` be an array or array view,
/// let `B` be an array with owned data,
/// and let `C` be an array with mutable data.
///
/// `ScalarOperand` determines for which scalars `K` operations `&A @ K`, and `B @ K`,
/// and `C @= K` are defined, as ***right hand side operands***, for applicable
/// arithmetic operators (denoted `@`).
///
/// ***Left hand side*** scalar operands are not related to this trait
/// (they need one `impl` per concrete scalar type); but they are still
/// implemented for the same types, allowing operations
/// `K @ &A`, and `K @ B` for primitive numeric types `K`.
///
/// This trait ***does not*** limit which elements can be stored in an array in general.
/// Non-`ScalarOperand` types can still participate in arithmetic as array elements in
/// in array-array operations.
pub trait ScalarOperand : Any + Clone { }
impl ScalarOperand for bool { }
impl ScalarOperand for i8 { }
impl ScalarOperand for u8 { }
impl ScalarOperand for i16 { }
impl ScalarOperand for u16 { }
impl ScalarOperand for i32 { }
impl ScalarOperand for u32 { }
impl ScalarOperand for i64 { }
impl ScalarOperand for u64 { }
impl ScalarOperand for f32 { }
impl ScalarOperand for f64 { }
impl ScalarOperand for Complex<f32> { }
impl ScalarOperand for Complex<f64> { }

macro_rules! impl_binary_op(
    ($trt:ident, $mth:ident, $imth:ident, $imth_scalar:ident, $doc:expr) => (
/// Perform elementwise
#[doc=$doc]
/// between `self` and `rhs`,
/// and return the result (based on `self`).
///
/// `self` must be an `OwnedArray` or `RcArray`.
///
/// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
///
/// **Panics** if broadcasting isn’t possible.
impl<A, S, S2, D, E> $trt<ArrayBase<S2, E>> for ArrayBase<S, D>
    where A: Clone + $trt<A, Output=A>,
          S: DataOwned<Elem=A> + DataMut,
          S2: Data<Elem=A>,
          D: Dimension,
          E: Dimension,
{
    type Output = ArrayBase<S, D>;
    fn $mth(self, rhs: ArrayBase<S2, E>) -> ArrayBase<S, D>
    {
        self.$mth(&rhs)
    }
}

/// Perform elementwise
#[doc=$doc]
/// between `self` and reference `rhs`,
/// and return the result (based on `self`).
///
/// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
///
/// **Panics** if broadcasting isn’t possible.
impl<'a, A, S, S2, D, E> $trt<&'a ArrayBase<S2, E>> for ArrayBase<S, D>
    where A: Clone + $trt<A, Output=A>,
          S: DataMut<Elem=A>,
          S2: Data<Elem=A>,
          D: Dimension,
          E: Dimension,
{
    type Output = ArrayBase<S, D>;
    fn $mth (mut self, rhs: &ArrayBase<S2, E>) -> ArrayBase<S, D>
    {
        self.$imth(rhs);
        self
    }
}

/// Perform elementwise
#[doc=$doc]
/// between references `self` and `rhs`,
/// and return the result as a new `OwnedArray`.
///
/// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
///
/// **Panics** if broadcasting isn’t possible.
impl<'a, 'b, A, S, S2, D, E> $trt<&'a ArrayBase<S2, E>> for &'b ArrayBase<S, D>
    where A: Clone + $trt<A, Output=A>,
          S: Data<Elem=A>,
          S2: Data<Elem=A>,
          D: Dimension,
          E: Dimension,
{
    type Output = OwnedArray<A, D>;
    fn $mth (self, rhs: &'a ArrayBase<S2, E>) -> OwnedArray<A, D>
    {
        // FIXME: Can we co-broadcast arrays here? And how?
        self.to_owned().$mth(rhs)
    }
}

/// Perform elementwise
#[doc=$doc]
/// between `self` and the scalar `x`,
/// and return the result (based on `self`).
///
/// `self` must be an `OwnedArray` or `RcArray`.
impl<A, S, D, B> $trt<B> for ArrayBase<S, D>
    where A: Clone + $trt<B, Output=A>,
          S: DataOwned<Elem=A> + DataMut,
          D: Dimension,
          B: ScalarOperand,
{
    type Output = ArrayBase<S, D>;
    fn $mth (mut self, x: B) -> ArrayBase<S, D>
    {
        self.unordered_foreach_mut(move |elt| {
            *elt = elt.clone().$mth(x.clone());
        });
        self
    }
}

/// Perform elementwise
#[doc=$doc]
/// between the reference `self` and the scalar `x`,
/// and return the result as a new `OwnedArray`.
impl<'a, A, S, D, B> $trt<B> for &'a ArrayBase<S, D>
    where A: Clone + $trt<B, Output=A>,
          S: Data<Elem=A>,
          D: Dimension,
          B: ScalarOperand,
{
    type Output = OwnedArray<A, D>;
    fn $mth(self, x: B) -> OwnedArray<A, D>
    {
        self.to_owned().$mth(x)
    }
}
    );
);

macro_rules! impl_scalar_op {
    ($scalar:ty, $trt:ident, $mth:ident, $doc:expr) => (
// these have no doc -- they are not visible in rustdoc
// Perform elementwise
// between the scalar `self` and array `rhs`,
// and return the result (based on `self`).
impl<S, D> $trt<ArrayBase<S, D>> for $scalar
    where S: DataMut<Elem=$scalar>,
          D: Dimension,
{
    type Output = ArrayBase<S, D>;
    fn $mth (self, mut rhs: ArrayBase<S, D>) -> ArrayBase<S, D>
    {
        rhs.unordered_foreach_mut(move |elt| {
            *elt = self.$mth(*elt);
        });
        rhs
    }
}

// Perform elementwise
// between the scalar `self` and array `rhs`,
// and return the result as a new `OwnedArray`.
impl<'a, S, D> $trt<&'a ArrayBase<S, D>> for $scalar
    where S: Data<Elem=$scalar>,
          D: Dimension,
{
    type Output = OwnedArray<$scalar, D>;
    fn $mth (self, rhs: &ArrayBase<S, D>) -> OwnedArray<$scalar, D>
    {
        self.$mth(rhs.to_owned())
    }
}
    );
}


mod arithmetic_ops {
    use super::*;
    use imp_prelude::*;

    use std::ops::*;
    use libnum::Complex;

    impl_binary_op!(Add, add, iadd, iadd_scalar, "addition");
    impl_binary_op!(Sub, sub, isub, isub_scalar, "subtraction");
    impl_binary_op!(Mul, mul, imul, imul_scalar, "multiplication");
    impl_binary_op!(Div, div, idiv, idiv_scalar, "division");
    impl_binary_op!(Rem, rem, irem, irem_scalar, "remainder");
    impl_binary_op!(BitAnd, bitand, ibitand, ibitand_scalar, "bit and");
    impl_binary_op!(BitOr, bitor, ibitor, ibitor_scalar, "bit or");
    impl_binary_op!(BitXor, bitxor, ibitxor, ibitxor_scalar, "bit xor");
    impl_binary_op!(Shl, shl, ishl, ishl_scalar, "left shift");
    impl_binary_op!(Shr, shr, ishr, ishr_scalar, "right shift");

    macro_rules! all_scalar_ops {
        ($int_scalar:ty) => (
            impl_scalar_op!($int_scalar, Add, add, "addition");
            impl_scalar_op!($int_scalar, Sub, sub, "subtraction");
            impl_scalar_op!($int_scalar, Mul, mul, "multiplication");
            impl_scalar_op!($int_scalar, Div, div, "division");
            impl_scalar_op!($int_scalar, Rem, rem, "remainder");
            impl_scalar_op!($int_scalar, BitAnd, bitand, "bit and");
            impl_scalar_op!($int_scalar, BitOr, bitor, "bit or");
            impl_scalar_op!($int_scalar, BitXor, bitxor, "bit xor");
            impl_scalar_op!($int_scalar, Shl, shl, "left shift");
            impl_scalar_op!($int_scalar, Shr, shr, "right shift");
        );
    }
    all_scalar_ops!(i8);
    all_scalar_ops!(u8);
    all_scalar_ops!(i16);
    all_scalar_ops!(u16);
    all_scalar_ops!(i32);
    all_scalar_ops!(u32);
    all_scalar_ops!(i64);
    all_scalar_ops!(u64);

    impl_scalar_op!(bool, BitAnd, bitand, "bit and");
    impl_scalar_op!(bool, BitOr, bitor, "bit or");
    impl_scalar_op!(bool, BitXor, bitxor, "bit xor");

    impl_scalar_op!(f32, Add, add, "addition");
    impl_scalar_op!(f32, Sub, sub, "subtraction");
    impl_scalar_op!(f32, Mul, mul, "multiplication");
    impl_scalar_op!(f32, Div, div, "division");
    impl_scalar_op!(f32, Rem, rem, "remainder");

    impl_scalar_op!(f64, Add, add, "addition");
    impl_scalar_op!(f64, Sub, sub, "subtraction");
    impl_scalar_op!(f64, Mul, mul, "multiplication");
    impl_scalar_op!(f64, Div, div, "division");
    impl_scalar_op!(f64, Rem, rem, "remainder");

    impl_scalar_op!(Complex<f32>, Add, add, "addition");
    impl_scalar_op!(Complex<f32>, Sub, sub, "subtraction");
    impl_scalar_op!(Complex<f32>, Mul, mul, "multiplication");
    impl_scalar_op!(Complex<f32>, Div, div, "division");

    impl_scalar_op!(Complex<f64>, Add, add, "addition");
    impl_scalar_op!(Complex<f64>, Sub, sub, "subtraction");
    impl_scalar_op!(Complex<f64>, Mul, mul, "multiplication");
    impl_scalar_op!(Complex<f64>, Div, div, "division");

    impl<A, S, D> Neg for ArrayBase<S, D>
        where A: Clone + Neg<Output=A>,
              S: DataMut<Elem=A>,
              D: Dimension
    {
        type Output = Self;
        /// Perform an elementwise negation of `self` and return the result.
        fn neg(mut self) -> Self {
            self.ineg();
            self
        }
    }

    impl<A, S, D> Not for ArrayBase<S, D>
        where A: Clone + Not<Output=A>,
              S: DataMut<Elem=A>,
              D: Dimension
    {
        type Output = Self;
        /// Perform an elementwise unary not of `self` and return the result.
        fn not(mut self) -> Self {
            self.inot();
            self
        }
    }
}

#[cfg(feature = "assign_ops")]
mod assign_ops {
    use super::*;
    use imp_prelude::*;

    macro_rules! impl_assign_op {
        ($trt:ident, $method:ident, $doc:expr) => {
    use std::ops::$trt;

    #[doc=$doc]
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting isn’t possible.
    ///
    /// **Requires crate feature `"assign_ops"`**
    impl<'a, A, S, S2, D, E> $trt<&'a ArrayBase<S2, E>> for ArrayBase<S, D>
        where A: Clone + $trt<A>,
              S: DataMut<Elem=A>,
              S2: Data<Elem=A>,
              D: Dimension,
              E: Dimension,
    {
        fn $method(&mut self, rhs: &ArrayBase<S2, E>) {
            self.zip_mut_with(rhs, |x, y| {
                x.$method(y.clone());
            });
        }
    }

    #[doc=$doc]
    /// **Requires crate feature `"assign_ops"`**
    impl<A, S, D> $trt<A> for ArrayBase<S, D>
        where A: ScalarOperand + $trt<A>,
              S: DataMut<Elem=A>,
              D: Dimension,
    {
        fn $method(&mut self, rhs: A) {
            self.unordered_foreach_mut(move |elt| {
                elt.$method(rhs.clone());
            });
        }
    }

        };
    }

    impl_assign_op!(AddAssign, add_assign,
                    "Perform `self += rhs` as elementwise addition (in place).\n");
    impl_assign_op!(SubAssign, sub_assign,
                    "Perform `self -= rhs` as elementwise subtraction (in place).\n");
    impl_assign_op!(MulAssign, mul_assign,
                    "Perform `self *= rhs` as elementwise multiplication (in place).\n");
    impl_assign_op!(DivAssign, div_assign,
                    "Perform `self /= rhs` as elementwise division (in place).\n");
    impl_assign_op!(RemAssign, rem_assign,
                    "Perform `self %= rhs` as elementwise remainder (in place).\n");
    impl_assign_op!(BitAndAssign, bitand_assign,
                    "Perform `self &= rhs` as elementwise bit and (in place).\n");
    impl_assign_op!(BitOrAssign, bitor_assign,
                    "Perform `self |= rhs` as elementwise bit or (in place).\n");
    impl_assign_op!(BitXorAssign, bitxor_assign,
                    "Perform `self ^= rhs` as elementwise bit xor (in place).\n");
    impl_assign_op!(ShlAssign, shl_assign,
                    "Perform `self <<= rhs` as elementwise left shift (in place).\n");
    impl_assign_op!(ShrAssign, shr_assign,
                    "Perform `self >>= rhs` as elementwise right shift (in place).\n");
}
