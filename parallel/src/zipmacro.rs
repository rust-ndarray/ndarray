// Copyright 2017 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_export]
/// Parallel version of the `azip!` macro.
///
/// See the `azip!` documentation for more details.
///
/// This example:
///
/// ```rust,ignore
/// par_azip!(mut a, b, c in { *a = b + c })
/// ```
///
/// Is equivalent to:
///
/// ```rust,ignore
/// Zip::from(&mut a).and(&b).and(&c).par_apply(|a, &b, &c| {
///     *a = b + c;
/// });
/// ```
///
/// **Panics** if any of the arrays are not of the same shape.
///
/// ## Examples
///
/// ```rust
/// extern crate ndarray;
/// #[macro_use(par_azip)]
/// extern crate ndarray_parallel;
///
/// use ndarray::Array2;
///
/// type M = Array2<f32>;
///
/// fn main() {
///     let mut a = M::zeros((16, 16));
///     let b = M::from_elem(a.dim(), 1.);
///     let c = M::from_elem(a.dim(), 2.);
///
///     // Compute a simple ternary operation:
///     // elementwise addition of b and c, stored in a
///
///     par_azip!(mut a, b, c in { *a = b + c });
///
///     assert_eq!(a, &b + &c);
/// }
/// ```
macro_rules! par_azip {
    // Build Zip Rule (index)
    (@parse [index => $a:expr, $($aa:expr,)*] $t1:tt in $t2:tt) => {
        par_azip!(@finish ($crate::ndarray::Zip::indexed($a)) [$($aa,)*] $t1 in $t2)
    };
    // Build Zip Rule (no index)
    (@parse [$a:expr, $($aa:expr,)*] $t1:tt in $t2:tt) => {
        par_azip!(@finish ($crate::ndarray::Zip::from($a)) [$($aa,)*] $t1 in $t2)
    };
    // Build Finish Rule (both)
    (@finish ($z:expr) [$($aa:expr,)*] [$($p:pat,)+] in { $($t:tt)*}) => {
        use $crate::prelude::*;
        #[allow(unused_mut)]
        ($z)
            $(
                .and($aa)
            )*
            .par_apply(|$($p),+| {
                $($t)*
            })
    };
    // parsing stack: [expressions] [patterns] (one per operand)
    // index uses empty [] -- must be first
    (@parse [] [] index $i:pat, $($t:tt)*) => {
        par_azip!(@parse [index =>] [$i,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] mut $x:ident ($e:expr) $($t:tt)*) => {
        par_azip!(@parse [$($exprs)* $e,] [$($pats)* mut $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] mut $x:ident $($t:tt)*) => {
        par_azip!(@parse [$($exprs)* &mut $x,] [$($pats)* mut $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] , $($t:tt)*) => {
        par_azip!(@parse [$($exprs)*] [$($pats)*] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] ref $x:ident ($e:expr) $($t:tt)*) => {
        par_azip!(@parse [$($exprs)* $e,] [$($pats)* $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] ref $x:ident $($t:tt)*) => {
        par_azip!(@parse [$($exprs)* &$x,] [$($pats)* $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] $x:ident ($e:expr) $($t:tt)*) => {
        par_azip!(@parse [$($exprs)* $e,] [$($pats)* &$x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] $x:ident $($t:tt)*) => {
        par_azip!(@parse [$($exprs)* &$x,] [$($pats)* &$x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] $($t:tt)*) => { };
    ($($t:tt)*) => {
        par_azip!(@parse [] [] $($t)*);
    }
}
