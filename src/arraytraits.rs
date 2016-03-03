#[cfg(feature = "rustc-serialize")]
use serialize::{Encodable, Encoder, Decodable, Decoder};

use std::hash;
use std::iter::FromIterator;
use std::iter::IntoIterator;
use std::ops::{
    Index,
    IndexMut,
};

use super::{
    Dimension, Ix,
    Elements,
    ElementsMut,
    ArrayBase,
    ArrayView,
    ArrayViewMut,
    Data,
    DataMut,
    DataOwned,
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
pub fn debug_bounds_check<S, D, I>(a: &ArrayBase<S, D>, index: &I)
    where D: Dimension,
          I: NdIndex<Dim=D>,
          S: Data,
{
    debug_bounds_check!(a, *index);
}

/// Access the element at **index**.
///
/// **Panics** if index is out of bounds.
impl<S, D, I> Index<I> for ArrayBase<S, D>
    where D: Dimension,
          I: NdIndex<Dim=D>,
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
          I: NdIndex<Dim=D>,
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

impl<A, S> FromIterator<A> for ArrayBase<S, Ix>
    where S: DataOwned<Elem=A>
{
    fn from_iter<I>(iterable: I) -> ArrayBase<S, Ix>
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


#[cfg(feature = "rustc-serialize")]
// Use version number so we can add a packed format later.
static ARRAY_FORMAT_VERSION: u8 = 1u8;

/// **Requires crate feature `"rustc-serialize"`**
#[cfg(feature = "rustc-serialize")]
impl<A, S, D> Encodable for ArrayBase<S, D>
    where A: Encodable,
          D: Dimension + Encodable,
          S: Data<Elem = A>
{
    fn encode<E: Encoder>(&self, s: &mut E) -> Result<(), E::Error> {
        s.emit_struct("Array", 3, |e| {
            try!(e.emit_struct_field("v", 0, |e| ARRAY_FORMAT_VERSION.encode(e)));
            // FIXME: Write self.dim as a slice (self.shape)
            // The problem is decoding it.
            try!(e.emit_struct_field("dim", 1, |e| self.dim.encode(e)));
            try!(e.emit_struct_field("data", 2, |e| {
                let sz = self.dim.size();
                e.emit_seq(sz, |e| {
                    for (i, elt) in self.iter().enumerate() {
                        try!(e.emit_seq_elt(i, |e| elt.encode(e)))
                    }
                    Ok(())
                })
            }));
            Ok(())
        })
    }
}

/// **Requires crate feature `"rustc-serialize"`**
#[cfg(feature = "rustc-serialize")]
impl<A, S, D> Decodable for ArrayBase<S, D>
    where A: Decodable,
          D: Dimension + Decodable,
          S: DataOwned<Elem = A>
{
    fn decode<E: Decoder>(d: &mut E) -> Result<ArrayBase<S, D>, E::Error> {
        d.read_struct("Array", 3, |d| {
            let version: u8 = try!(d.read_struct_field("v", 0, Decodable::decode));
            if version > ARRAY_FORMAT_VERSION {
                return Err(d.error("unknown array version"))
            }
            let dim: D = try!(d.read_struct_field("dim", 1, |d| {
                Decodable::decode(d)
            }));
            let elements = try!(
                d.read_struct_field("data", 2, |d| {
                    d.read_seq(|d, len| {
                        if len != dim.size() {
                            Err(d.error("data and dimension must match in size"))
                        } else {
                            let mut elements = Vec::with_capacity(len);
                            for i in 0..len {
                                elements.push(try!(d.read_seq_elt::<A, _>(i, Decodable::decode)))
                            }
                            Ok(elements)
                        }
                    })
            }));
            unsafe {
                Ok(ArrayBase::from_vec_dim_unchecked(dim, elements))
            }
        })
    }
}


/// Create a one-dimensional read-only array view of the data in `slice`.
impl<'a, A, Slice: ?Sized> From<&'a Slice> for ArrayView<'a, A, Ix>
    where Slice: AsRef<[A]>
{
    fn from(slice: &'a Slice) -> Self {
        let xs = slice.as_ref();
        unsafe {
            Self::new_(xs.as_ptr(), xs.len(), 1)
        }
    }
}

impl<'a, A, S, D> From<&'a ArrayBase<S, D>> for ArrayView<'a, A, D>
    where S: Data<Elem=A>,
          D: Dimension,
{
    fn from(array: &'a ArrayBase<S, D>) -> Self {
        array.view()
    }
}

/// Create a one-dimensional read-write array view of the data in `slice`.
impl<'a, A, Slice: ?Sized> From<&'a mut Slice> for ArrayViewMut<'a, A, Ix>
    where Slice: AsMut<[A]>
{
    fn from(slice: &'a mut Slice) -> Self {
        let xs = slice.as_mut();
        unsafe {
            Self::new_(xs.as_mut_ptr(), xs.len(), 1)
        }
    }
}

impl<'a, A, S, D> From<&'a mut ArrayBase<S, D>> for ArrayViewMut<'a, A, D>
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
pub trait AsArray<'a, A: 'a, D = Ix> : Into<ArrayView<'a, A, D>> where D: Dimension { }
impl<'a, A: 'a, D, T> AsArray<'a, A, D> for T
    where T: Into<ArrayView<'a, A, D>>,
          D: Dimension,
{ }
