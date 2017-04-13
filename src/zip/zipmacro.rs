
#[macro_export]
/// Array zip macro: lock step function application across several arrays and
/// producers.
///
/// This is a shorthand for [`Zip`](struct.Zip.html).
///
/// This example:
///
/// ```rust,ignore
/// azip!(mut a, b, c in { *a = b + c })
/// ```
///
/// Is equivalent to:
///
/// ```rust,ignore
/// Zip::from(&mut a).and(&b).and(&c).apply(|a, &b, &c| {
///     *a = b + c;
/// });
///
/// ```
///
/// Explanation of the shorthand for captures:
///
/// + `mut a`: the producer is `&mut a` and the variable pattern is `mut a`.
/// + `b`: the producer is `&b` and the variable pattern is `&b` (same for `c`).
///
/// The syntax is `azip!(` *capture [, capture [, ...] ]* `in {` *expression* `})`
/// where the captures are a sequence of pattern-like items that indicate which
/// arrays are used for the zip. The *expression* is evaluated elementwise,
/// with the value of an element from each producer in their respective variable.
///
/// More capture rules:
///
/// + `ref c`: the producer is `&c` and the variable pattern is `c`.
/// + `mut a (expr)`: the producer is `expr` and the variable pattern is `mut a`.
/// + `b (expr)`: the producer is `expr` and the variable pattern is `&b`.
/// + `ref c (expr)`: the producer is `expr` and the variable pattern is `c`.
///
/// **Panics** if any of the arrays are not of the same shape.
///
/// ## Examples
///
/// ```rust
/// #[macro_use(azip)]
/// extern crate ndarray;
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
///     azip!(mut a, b, c in { *a = b + c });
///
///     assert_eq!(a, &b + &c);
/// }
/// ```
macro_rules! azip {
    // Final Rule
    (@parse [$a:expr, $($aa:expr,)*] [$($p:pat,)+] in { $($t:tt)* }) => {
        $crate::Zip::from($a)
            $(
                .and($aa)
            )*
            .apply(|$($p),+| {
                $($t)*
            })
    };
    // parsing stack: [expressions] [patterns] (one per operand)
    (@parse [$($exprs:tt)*] [$($pats:tt)*] mut $x:ident ($e:expr) $($t:tt)*) => {
        azip!(@parse [$($exprs)* $e,] [$($pats)* mut $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] mut $x:ident $($t:tt)*) => {
        azip!(@parse [$($exprs)* &mut $x,] [$($pats)* mut $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] , $($t:tt)*) => {
        azip!(@parse [$($exprs)*] [$($pats)*] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] ref $x:ident ($e:expr) $($t:tt)*) => {
        azip!(@parse [$($exprs)* $e,] [$($pats)* $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] ref $x:ident $($t:tt)*) => {
        azip!(@parse [$($exprs)* &$x,] [$($pats)* $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] $x:ident ($e:expr) $($t:tt)*) => {
        azip!(@parse [$($exprs)* $e,] [$($pats)* &$x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] $x:ident $($t:tt)*) => {
        azip!(@parse [$($exprs)* &$x,] [$($pats)* &$x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] $($t:tt)*) => { };
    ($($t:tt)*) => {
        azip!(@parse [] [] $($t)*);
    }
}

