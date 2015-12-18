use std::ops::{Range, RangeFrom, RangeTo, RangeFull};
use super::{Ixs};

// [a:b:s] syntax for example [:3], [::-1]
// [0,:] -- first row of matrix
// [:,0] -- first column of matrix

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// A slice, a description of a range of an array axis.
///
/// Fields are `begin`, `end` and `stride`, where
/// negative `begin` or `end` indexes are counted from the back
/// of the axis.
///
/// If `end` is `None`, the slice extends to the end of the axis.
///
/// ## Examples
///
/// `Si(0, None, 1)` is the full range of an axis.
/// Python equivalent is `[:]`.
///
/// `Si(a, Some(b), 2)` is every second element from `a` until `b`.
/// Python equivalent is `[a:b:2]`.
///
/// `Si(a, None, -1)` is every element, in reverse order, from `a`
/// until the end. Python equivalent is `[a::-1]`
pub struct Si(pub Ixs, pub Option<Ixs>, pub Ixs);

impl From<Range<Ixs>> for Si
{
    #[inline]
    fn from(r: Range<Ixs>) -> Si { Si(r.start, Some(r.end), 1) }
}

impl From<RangeFrom<Ixs>> for Si
{
    #[inline]
    fn from(r: RangeFrom<Ixs>) -> Si { Si(r.start, None, 1) }
}

impl From<RangeTo<Ixs>> for Si
{
    #[inline]
    fn from(r: RangeTo<Ixs>) -> Si { Si(0, Some(r.end), 1) }
}

impl From<RangeFull> for Si
{
    #[inline]
    fn from(_: RangeFull) -> Si { S }
}


impl Si {
    #[inline]
    pub fn step(self, step: Ixs) -> Self {
        Si(self.0, self.1, self.2 * step)
    }
}

/// Slice value for the full range of an axis.
pub const S: Si = Si(0, None, 1);

/// Slice argument constructor.
///
/// `s![]` takes a list of ranges, separated by comma, with optional strides
/// that are separated from the range by a semicolon.
/// It is converted into a slice argument with type `&[Si; n]`.
///
/// For example `s![a..b;c, d..e]`
/// is equivalent to `&[Si(a, Some(b), c), Si(d, Some(e), 1)]`.
///
/// ```
/// #[macro_use]
/// extern crate ndarray;
///
/// use ndarray::{
///     ArrayView,
///     aview0,
///     Ix,
///     OwnedArray,
/// };
///
/// fn laplacian(v: &ArrayView<f32, (Ix, Ix)>) -> OwnedArray<f32, (Ix, Ix)> {
///     (&v.slice(s![1..-1, 1..-1]) * &aview0(&-4.))
///     + v.slice(s![ ..-2, 1..-1])
///     + v.slice(s![1..-1,  ..-2])
///     + v.slice(s![1..-1, 2..  ])
///     + v.slice(s![2..  , 1..-1])
/// }
/// # fn main() { }
/// ```
#[macro_export]
macro_rules! s(
    (@as_expr $e:expr) => ($e);
    (@parse [$($stack:tt)*] $r:expr;$s:expr) => {
        s![@as_expr &[$($stack)* s!(@step $r, $s)]]
    };
    (@parse [$($stack:tt)*] $r:expr) => {
        s![@as_expr &[$($stack)* s!(@step $r, 1)]]
    };
    (@parse [$($stack:tt)*] $r:expr;$s:expr, $($t:tt)*) => {
        s![@parse [$($stack)* s!(@step $r, $s),] $($t)*]
    };
    (@parse [$($stack:tt)*] $r:expr, $($t:tt)*) => {
        s![@parse [$($stack)* s!(@step $r, 1),] $($t)*]
    };
    (@step $r:expr, $s:expr) => {
        <$crate::Si as ::std::convert::From<_>>::from($r).step($s)
    };
    ($($t:tt)*) => {
        s![@parse [] $($t)*]
    };
);
