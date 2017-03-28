
#[macro_export]
/// Array zip macro.
///
/// This is a shorthand for lock step iteration across many arrays.
///
/// This example:
///
/// ```rust,ignore
/// array_zip!(mut a, b, c in { *a = b + c })
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
/// + `mut a`: the array is `&mut a` and the variable pattern is `a`.
/// + `b`: the array is `&b` and the variable pattern is `&b` (same for `c`).
///
/// The syntax is `array_zip!(` *capture [, capture [, ...] ]* `in {` *expression* `})`
/// where the captures are a sequence of pattern-like items that indicate which
/// arrays are used for the zip. The *expression* is evaluated elementwise,
/// with the value of an element from each array in their respective variable.
///
/// More capture rules:
///
/// + `ref c`: the array is `&c` and the variable pattern is `c`.
/// + `mut a (expr)`: the array is `expr` and the variable pattern is `a`.
/// + `b (expr)`: the array is `expr` and the variable pattern is `&b`.
/// + `ref c (expr)`: the array is `expr` and the variable pattern is `c`.
///
/// **Panics** if any of the arrays are not of the same shape.
///
/// ## Examples
///
/// ```rust
/// #[macro_use(array_zip)]
/// extern crate ndarray;
///
/// use ndarray::Array2;
///
/// type M = Array2<f32>;
///
/// # fn main() {
///
/// let mut a = M::zeros((16, 16));
/// let b = M::from_elem(a.dim(), 1.);
/// let c = M::from_elem(a.dim(), 2.);
///
/// array_zip!(mut a, b, c in { *a = b + c });
/// # }
/// ```
macro_rules! array_zip {
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
        array_zip!(@parse [$($exprs)* $e,] [$($pats)* mut $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] mut $x:ident $($t:tt)*) => {
        array_zip!(@parse [$($exprs)* &mut $x,] [$($pats)* mut $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] , $($t:tt)*) => {
        array_zip!(@parse [$($exprs)*] [$($pats)*] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] ref $x:ident ($e:expr) $($t:tt)*) => {
        array_zip!(@parse [$($exprs)* $e,] [$($pats)* $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] ref $x:ident $($t:tt)*) => {
        array_zip!(@parse [$($exprs)* &$x,] [$($pats)* $x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] $x:ident ($e:expr) $($t:tt)*) => {
        array_zip!(@parse [$($exprs)* $e,] [$($pats)* &$x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] $x:ident $($t:tt)*) => {
        array_zip!(@parse [$($exprs)* &$x,] [$($pats)* &$x,] $($t)*);
    };
    (@parse [$($exprs:tt)*] [$($pats:tt)*] $($t:tt)*) => { };
    ($($t:tt)*) => {
        array_zip!(@parse [] [] $($t)*);
    }
}

