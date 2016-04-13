// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::{
    Add, Sub, Mul, Div, Rem, Neg, Not, Shr, Shl,
    BitAnd,
    BitOr,
    BitXor,
};
use imp_prelude::*;
// array OPERATORS

macro_rules! as_expr(
    ($e:expr) => ($e)
);

macro_rules! impl_binary_op_inplace(
    ($trt:ident, $operator:tt, $mth:ident, $imethod:ident, $imth_scalar:ident, $doc:expr) => (
    #[cfg_attr(all(has_deprecated, has_assign), deprecated(note=
      "Use augmented assignments (+= and similar) instead."))]
    /// Perform elementwise
    #[doc=$doc]
    /// between `self` and `rhs`,
    /// *in place*.
    ///
    /// If their shapes disagree, `rhs` is broadcast to the shape of `self`.
    ///
    /// **Panics** if broadcasting isn’t possible.
    pub fn $imethod <E: Dimension, S2> (&mut self, rhs: &ArrayBase<S2, E>)
        where A: Clone + $trt<A, Output=A>,
              S2: Data<Elem=A>,
    {
        self.zip_mut_with(rhs, |x, y| {
            *x = as_expr!(x.clone() $operator y.clone());
        });
    }

    #[cfg_attr(all(has_deprecated, has_assign), deprecated(note=
      "Use augmented assignments (+= and similar) instead."))]
    /// Perform elementwise
    #[doc=$doc]
    /// between `self` and the scalar `x`,
    /// *in place*.
    pub fn $imth_scalar (&mut self, x: &A)
        where A: Clone + $trt<A, Output=A>,
    {
        self.unordered_foreach_mut(move |elt| {
            *elt = as_expr!(elt.clone() $operator x.clone());
        });
    }
    );
);

/// In-place arithmetic operations.
///
/// ***Note: These will be deprecated in favour of overloaded `@=` operators
/// when Rust 1.8 is released.***
impl<A, S, D> ArrayBase<S, D>
    where S: DataMut<Elem=A>,
          D: Dimension,
{


impl_binary_op_inplace!(Add, +, add, iadd, iadd_scalar, "addition");
impl_binary_op_inplace!(Sub, -, sub, isub, isub_scalar, "subtraction");
impl_binary_op_inplace!(Mul, *, mul, imul, imul_scalar, "multiplication");
impl_binary_op_inplace!(Div, /, div, idiv, idiv_scalar, "division");
impl_binary_op_inplace!(Rem, %, rem, irem, irem_scalar, "remainder");
impl_binary_op_inplace!(BitAnd, &, bitand, ibitand, ibitand_scalar, "bit and");
impl_binary_op_inplace!(BitOr, |, bitor, ibitor, ibitor_scalar, "bit or");
impl_binary_op_inplace!(BitXor, ^, bitxor, ibitxor, ibitxor_scalar, "bit xor");
impl_binary_op_inplace!(Shl, <<, shl, ishl, ishl_scalar, "left shift");
impl_binary_op_inplace!(Shr, >>, shr, ishr, ishr_scalar, "right shift");

    #[cfg_attr(all(has_deprecated, has_assign), deprecated(note=
      "Use augmented assignments (+= and similar) instead."))]
    /// Perform an elementwise negation of `self`, *in place*.
    pub fn ineg(&mut self)
        where A: Clone + Neg<Output=A>,
    {
        self.unordered_foreach_mut(|elt| {
            *elt = -elt.clone();
        });
    }

    #[cfg_attr(all(has_deprecated, has_assign), deprecated(note=
      "Use augmented assignments (+= and similar) instead."))]
    /// Perform an elementwise unary not of `self`, *in place*.
    pub fn inot(&mut self)
        where A: Clone + Not<Output=A>,
    {
        self.unordered_foreach_mut(|elt| {
            *elt = !elt.clone();
        });
    }

}

