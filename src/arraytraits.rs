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
    Array, Dimension, Ix, Elements, ElementsMut,
    ArrayBase,
    ArrayView,
    ArrayViewMut,
    Data,
    DataMut,
};

/// Access the element at **index**.
///
/// **Panics** if index is out of bounds.
impl<S, D> Index<D> for ArrayBase<S, D>
    where D: Dimension,
          S: Data,
{
    type Output = S::Elem;
    #[inline]
    fn index(&self, index: D) -> &S::Elem {
        self.get(index).expect("Array::index: out of bounds")
    }
}

/// Access the element at **index** mutably.
///
/// **Panics** if index is out of bounds.
impl<S, D> IndexMut<D> for ArrayBase<S, D>
    where D: Dimension,
          S: DataMut,
{
    #[inline]
    fn index_mut(&mut self, index: D) -> &mut S::Elem {
        self.get_mut(index).expect("Array::index_mut: out of bounds")
    }
}


impl<S, S2, D> PartialEq<ArrayBase<S2, D>> for ArrayBase<S, D>
    where D: Dimension,
          S: Data,
          S2: Data<Elem = S::Elem>,
          S::Elem: PartialEq,
{
    /// Return `true` if the array shapes and all elements of `self` and
    /// `other` are equal. Return `false` otherwise.
    fn eq(&self, other: &ArrayBase<S2, D>) -> bool
    {
        self.shape() == other.shape() &&
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<S, D> Eq for ArrayBase<S, D>
    where D: Dimension,
          S: Data,
          S::Elem: Eq,
{ }

impl<A> FromIterator<A> for Array<A, Ix>
{
    fn from_iter<I: IntoIterator<Item=A>>(iterable: I) -> Array<A, Ix>
    {
        Array::from_iter(iterable)
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
          S: DataMut,
{
    type Item = &'a mut S::Elem;
    type IntoIter = ElementsMut<'a, S::Elem, D>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.iter_mut()
    }
}

impl<'a, A, D> IntoIterator for ArrayView<'a, A,  D>
    where D: Dimension,
{
    type Item = &'a A;
    type IntoIter = Elements<'a, A, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_iter_()
    }
}

impl<'a, A, D> IntoIterator for ArrayViewMut<'a, A,  D>
    where D: Dimension,
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
          S::Elem: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H)
    {
        self.shape().hash(state);
        for elt in self.iter() {
            elt.hash(state)
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

#[cfg(feature = "rustc-serialize")]
impl<A: Encodable, D: Dimension + Encodable> Encodable for Array<A, D>
{
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error>
    {
        s.emit_struct("Array", 3, |e| {
            try!(e.emit_struct_field("v", 0, |e| {
                ARRAY_FORMAT_VERSION.encode(e)
            }));
            // FIXME: Write self.dim as a slice (self.shape)
            // The problem is decoding it.
            try!(e.emit_struct_field("dim", 1,
                                           |e| self.dim.encode(e)));
            try!(e.emit_struct_field("data", 2, |e| {
                let sz = self.dim.size();
                e.emit_seq(sz, |e| {
                    for (i, elt) in self.iter().enumerate() {
                        try!(e.emit_seq_elt(i, |e| {
                            elt.encode(e)
                        }))
                    }
                    Ok(())
                })
            }));
            Ok(())
        })
    }
}

#[cfg(feature = "rustc-serialize")]
impl<A: Decodable, D: Dimension + Decodable>
    Decodable for Array<A, D>
{
    fn decode<S: Decoder>(d: &mut S) -> Result<Array<A, D>, S::Error>
    {
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
                            for i in (0..len) {
                                elements.push(try!(d.read_seq_elt::<A, _>(i, Decodable::decode)))
                            }
                            Ok(elements)
                        }
                    })
            }));
            unsafe {
                Ok(Array::from_vec_dim(dim, elements))
            }
        })
    }
}
