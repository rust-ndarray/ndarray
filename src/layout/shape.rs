use core::iter::Map;

use crate::layout::ranked::Ranked;

/// A trait for array shapes: lists of `usize` describing the length of each dimension of an array.
///
/// ## Size Limits
/// Arrays cannot have more than [`usize::MAX`] elements; otherwise, it would not be possible
/// to address all of the elements in-memory. In addition, since an array may have negative strides
/// (i.e., elements _behind_ the current pointer in memory), arrays must be addressable using
/// [`isize`]. So no array can have more than [`isize::MAX`] elements. Finally, for multi-byte
/// element types, the total byte size of the array also cannot exceed `isize::MAX`.
///
/// Implementing `Shape` for a type does not guarantee that the type will always represent an
/// array that adheres to these size limits. It does, however, provide access to two methods that
/// can cheaply check these invariants: [`Shape::size_checked`] and [`Shape::size_bytes_checked`].
/// In order for an instance of a `Shape` to be valid, both of these methods must return `Some(_)`.
///
/// ## Mutability
/// The `Shape` trait does not provide any sort of mutability for the lengths of each axis.
/// This allows users to define constant-sized shapes, which can significantly increase performance.
///
/// Since `Shape` is still experimental, the mechanisms for mutability are still being designed.
pub trait Shape: Ranked
{
    /// The iterator type over the dimensions of the shape.
    type Iter<'a>: Iterator<Item = usize> + ExactSizeIterator + DoubleEndedIterator
    where Self: 'a;

    /// The length of the array along a given axis.
    fn axis_len(&self, axis: usize) -> usize;

    /// Iterate over the dimensions of the shape.
    fn iter(&self) -> Self::Iter<'_>;

    /// Get the number of elements that the array contains.
    ///
    /// If the number of elements is greater than `isize::MAX`, returns `None`.
    fn size_checked(&self) -> Option<usize>
    {
        self.iter()
            .try_fold(1_usize, |acc, i| acc.checked_mul(i))
            .and_then(as_usize_if_isize_compatible)
    }

    /// Get the number of bytes that this array would fill.
    ///
    /// This method checks for bytes overflow past `isize`. If this method returns `Some(_)`,
    /// then users know that an allocated array is indexable using `isize` offsets.
    fn size_bytes_checked<T>(&self) -> Option<usize>
    {
        self.size_checked()
            .and_then(|v| v.checked_mul(size_of::<T>()))
            .and_then(as_usize_if_isize_compatible)
    }

    /// Iterate over the shape as `isize`.
    ///
    /// If the number of elements is greater than `isize::MAX`, returns `None`.
    fn iter_isize<'a>(&'a self) -> Option<Map<Self::Iter<'a>, impl FnMut(usize) -> isize>>
    {
        self.size_checked().map(|_| self.iter().map(|v| v as isize))
    }
}

fn as_usize_if_isize_compatible(v: usize) -> Option<usize>
{
    if v <= (isize::MAX as usize) {
        Some(v)
    } else {
        None
    }
}
