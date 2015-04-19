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

pub trait SliceRange
{
    fn slice(self) -> Si;

    #[inline]
    fn step(self, step: Ixs) -> Si where
        Self: Sized,
    {
        self.slice().step(step)
    }
}

impl SliceRange for Range<Ixs>
{
    #[inline]
    fn slice(self) -> Si { Si(self.start, Some(self.end), 1) }
}

impl SliceRange for RangeFrom<Ixs>
{
    #[inline]
    fn slice(self) -> Si { Si(self.start, None, 1) }
}

impl SliceRange for RangeTo<Ixs>
{
    #[inline]
    fn slice(self) -> Si { Si(0, Some(self.end), 1) }
}

impl SliceRange for RangeFull
{
    #[inline]
    fn slice(self) -> Si { Si(0, None, 1) }
}


impl Si
{
    #[inline]
    pub fn from<R: SliceRange>(r: R) -> Self
    {
        r.slice()
    }

    #[inline]
    pub fn step(self, step: Ixs) -> Self
    {
        Si(self.0, self.1, self.2 * step)
    }
}

/// Slice value for the full range of an axis.
pub const S: Si = Si(0, None, 1);

