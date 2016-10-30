// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::hash;
use std::iter::FromIterator;
use std::iter::IntoIterator;
use std::ops::{
    Index,
    IndexMut,
};

use imp_prelude::*;
use {
    Elements,
    ElementsMut,
    NdIndex,
};

use numeric_util;

#[cold]
#[inline(never)]
fn array_out_of_bounds() -> ! {
    panic!("ndarray: index out of bounds");
}

// Macro to insert more informative out of bounds message in debug builds
#[cfg(debug_assertions)]
macro_rules! debug_bounds_check {
    ($self_:ident, $index:expr) => {
        if let None = $index.index_checked(&$self_.dim, &$self_.strides) {
            panic!("ndarray: index {:?} is out of bounds for array of shape {:?}",
                   $index, $self_.shape());
        }
    };
}

#[cfg(not(debug_assertions))]
macro_rules! debug_bounds_check {
    ($self_:ident, $index:expr) => { };
}

#[inline(always)]
pub fn debug_bounds_check<S, D, I>(_a: &ArrayBase<S, D>, _index: &I)
    where D: Dimension,
          I: NdIndex<D>,
          S: Data,
{
    debug_bounds_check!(_a, *_index);
}

/// Access the element at **index**.
///
/// **Panics** if index is out of bounds.
impl<S, D, I> Index<I> for ArrayBase<S, D>
    where D: Dimension,
          I: NdIndex<D>,
          S: Data,
{
    type Output = S::Elem;
    #[inline]
    fn index(&self, index: I) -> &S::Elem {
        debug_bounds_check!(self, index);
        self.get(index).unwrap_or_else(|| array_out_of_bounds())
    }
}

/// Access the element at **index** mutably.
///
/// **Panics** if index is out of bounds.
impl<S, D, I> IndexMut<I> for ArrayBase<S, D>
    where D: Dimension,
          I: NdIndex<D>,
          S: DataMut,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut S::Elem {
        debug_bounds_check!(self, index);
        self.get_mut(index).unwrap_or_else(|| array_out_of_bounds())
    }
}

/// Return `true` if the array shapes and all elements of `self` and
/// `rhs` are equal. Return `false` otherwise.
impl<S, S2, D> PartialEq<ArrayBase<S2, D>> for ArrayBase<S, D>
    where D: Dimension,
          S: Data,
          S2: Data<Elem = S::Elem>,
          S::Elem: PartialEq
{
    fn eq(&self, rhs: &ArrayBase<S2, D>) -> bool {
        if self.shape() != rhs.shape() {
            return false;
        }
        if let Some(self_s) = self.as_slice() {
            if let Some(rhs_s) = rhs.as_slice() {
                return numeric_util::unrolled_eq(self_s, rhs_s);
            }
        }
        self.iter().zip(rhs.iter()).all(|(a, b)| a == b)
    }
}

impl<S, D> Eq for ArrayBase<S, D>
    where D: Dimension,
          S: Data,
          S::Elem: Eq,
{ }

impl<A, S> FromIterator<A> for ArrayBase<S, Ix1>
    where S: DataOwned<Elem=A>
{
    fn from_iter<I>(iterable: I) -> ArrayBase<S, Ix1>
        where I: IntoIterator<Item=A>,
    {
        ArrayBase::from_iter(iterable)
    }
}

impl<'a, S, D> IntoIterator for &'a ArrayBase<S, D>
    where D: Dimension,
          S: Data,
{
    type Item = &'a S::Elem;
    type IntoIter = Elements<'a, S::Elem, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, S, D> IntoIterator for &'a mut ArrayBase<S, D>
    where D: Dimension,
          S: DataMut
{
    type Item = &'a mut S::Elem;
    type IntoIter = ElementsMut<'a, S::Elem, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, A, D> IntoIterator for ArrayView<'a, A, D>
    where D: Dimension
{
    type Item = &'a A;
    type IntoIter = Elements<'a, A, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_iter_()
    }
}

impl<'a, A, D> IntoIterator for ArrayViewMut<'a, A, D>
    where D: Dimension
{
    type Item = &'a mut A;
    type IntoIter = ElementsMut<'a, A, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_iter_()
    }
}

impl<'a, S, D> hash::Hash for ArrayBase<S, D>
    where D: Dimension,
          S: Data,
          S::Elem: hash::Hash
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.shape().hash(state);
        if let Some(self_s) = self.as_slice() {
            hash::Hash::hash_slice(self_s, state);
        } else {
            for row in self.inner_iter() {
                if let Some(row_s) = row.as_slice() {
                    hash::Hash::hash_slice(row_s, state);
                } else {
                    for elt in row {
                        elt.hash(state)
                    }
                }
            }
        }
    }
}

// NOTE: ArrayBase keeps an internal raw pointer that always
// points into the storage. This is Sync & Send as long as we
// follow the usual inherited mutability rules, as we do with
// Vec, &[] and &mut []

/// `ArrayBase` is `Sync` when the storage type is.
unsafe impl<S, D> Sync for ArrayBase<S, D>
    where S: Sync + Data, D: Sync
{ }

/// `ArrayBase` is `Send` when the storage type is.
unsafe impl<S, D> Send for ArrayBase<S, D>
    where S: Send + Data, D: Send
{ }

#[cfg(any(feature = "rustc-serialize", feature = "serde"))]
// Use version number so we can add a packed format later.
pub const ARRAY_FORMAT_VERSION: u8 = 1u8;


// use "raw" form instead of type aliases here so that they show up in docs
/// Implementation of `ArrayView::from(&S)` where `S` is a slice or slicable.
///
/// Create a one-dimensional read-only array view of the data in `slice`.
impl<'a, A, Slice: ?Sized> From<&'a Slice> for ArrayBase<ViewRepr<&'a A>, Ix1>
    where Slice: AsRef<[A]>
{
    fn from(slice: &'a Slice) -> Self {
        let xs = slice.as_ref();
        unsafe {
            Self::new_(xs.as_ptr(), Ix1(xs.len()), Ix1(1))
        }
    }
}

/// Implementation of `ArrayView::from(&A)` where `A` is an array.
///
/// Create a read-only array view of the array.
impl<'a, A, S, D> From<&'a ArrayBase<S, D>> for ArrayBase<ViewRepr<&'a A>, D>
    where S: Data<Elem=A>,
          D: Dimension,
{
    fn from(array: &'a ArrayBase<S, D>) -> Self {
        array.view()
    }
}

/// Implementation of `ArrayViewMut::from(&mut S)` where `S` is a slice or slicable.
///
/// Create a one-dimensional read-write array view of the data in `slice`.
impl<'a, A, Slice: ?Sized> From<&'a mut Slice> for ArrayBase<ViewRepr<&'a mut A>, Ix1>
    where Slice: AsMut<[A]>
{
    fn from(slice: &'a mut Slice) -> Self {
        let xs = slice.as_mut();
        unsafe {
            Self::new_(xs.as_mut_ptr(), Ix1(xs.len()), Ix1(1))
        }
    }
}

/// Implementation of `ArrayViewMut::from(&mut A)` where `A` is an array.
///
/// Create a read-write array view of the array.
impl<'a, A, S, D> From<&'a mut ArrayBase<S, D>> for ArrayBase<ViewRepr<&'a mut A>, D>
    where S: DataMut<Elem=A>,
          D: Dimension,
{
    fn from(array: &'a mut ArrayBase<S, D>) -> Self {
        array.view_mut()
    }
}

/// Argument conversion into an array view
///
/// The trait is parameterized over `A`, the element type, and `D`, the
/// dimensionality of the array. `D` defaults to one-dimensional.
///
/// Use `.into()` to do the conversion.
///
/// ```
/// use ndarray::AsArray;
///
/// fn sum<'a, V: AsArray<'a, f64>>(data: V) -> f64 {
///     let array_view = data.into();
///     array_view.scalar_sum()
/// }
///
/// assert_eq!(
///     sum(&[1., 2., 3.]),
///     6.
/// );
///
/// ```
pub trait AsArray<'a, A: 'a, D = Ix1> : Into<ArrayView<'a, A, D>> where D: Dimension { }
impl<'a, A: 'a, D, T> AsArray<'a, A, D> for T
    where T: Into<ArrayView<'a, A, D>>,
          D: Dimension,
{ }

/// Create an owned array with a default state.
///
/// The array is created with dimension `D::default()`, which results
/// in for example dimensions `0` and `(0, 0)` with zero elements for the
/// one-dimensional and two-dimensional cases respectively, while for example
/// the zero dimensional case uses `()` (or `Vec::new()`) which
/// results in an array with one element.
///
/// Since arrays cannot grow, the intention is to use the default value as
/// placeholder.
impl<A, S, D> Default for ArrayBase<S, D>
    where S: DataOwned<Elem=A>,
          D: Dimension,
          A: Default,
{
    // NOTE: We can implement Default for non-zero dimensional array views by
    // using an empty slice, however we need a trait for nonzero Dimension.
    fn default() -> Self {
        ArrayBase::default(D::default())
    }
}
