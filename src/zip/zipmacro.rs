
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
/// Special rule:
///
/// + `index i`: Use `Zip::indexed` instead. `i` is a pattern -- it can be
///    a single variable name or something else that pattern matches the index.
///    Must be first.
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
///     let mut b = M::zeros(a.dim());
///     let mut c = M::zeros(a.dim());
///
///     // set up values in b, c
///     b.fill(1.);
///     for ((i, j), elt) in c.indexed_iter_mut() {
///         *elt = (i + 10 * j) as f32;
///     }
///
///     // Compute a simple ternary operation:
///     // elementwise addition of b and c, stored in a
///
///     azip!(mut a, b, c in { *a = b + c });
///
///     assert_eq!(a, &b + &c);
///
///     // Example of azip!() with index
///
///     azip!(index (i, j), b, c in {
///         a[[i, j]] = b - c;
///     });
///
///     assert_eq!(a, &b - &c);
/// }
/// ```
macro_rules! azip {
    // Build Zip Rule (index)
    (@parse [index => $a:expr, $($aa:expr,)*] $t1:tt in $t2:tt) => {
        azip!(@finish ($crate::Zip::indexed($a)) [$($aa,)*] $t1 in $t2)
    };
    // Build Zip Rule (no index)
    (@parse [$a:expr, $($aa:expr,)*] $t1:tt in $t2:tt) => {
        azip!(@finish ($crate::Zip::from($a)) [$($aa,)*] $t1 in $t2)
    };
    // Build Finish Rule (both)
    (@finish ($z:expr) [$($aa:expr,)*] [$($p:pat,)+] in { $($t:tt)*}) => {
        ($z)
            $(
                .and($aa)
            )*
            .apply(|$($p),+| {
                $($t)*
            })
    };
    // parsing stack: [expressions] [patterns] (one per operand)
    // index uses empty [] -- must be first
    (@parse [] [] index $i:pat, $($t:tt)*) => {
        azip!(@parse [index =>] [$i,] $($t)*);
    };
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

