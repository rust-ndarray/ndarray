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
    Storage,
};

impl<'a, A, D: Dimension> Index<D> for Array<A, D>
{
    type Output = A;
    #[inline]
    /// Access the element at **index**.
    ///
    /// **Panics** if index is out of bounds.
    fn index(&self, index: D) -> &A {
        self.at(index).expect("Array::index: out of bounds")
    }
}

impl<'a, A: Clone, D: Dimension> IndexMut<D> for Array<A, D>
{
    #[inline]
    /// Access the element at **index** mutably.
    ///
    /// **Panics** if index is out of bounds.
    fn index_mut(&mut self, index: D) -> &mut A {
        self.at_mut(index).expect("Array::index_mut: out of bounds")
    }
}


impl<S, S2, D> PartialEq<ArrayBase<S2, D>> for ArrayBase<S, D>
    where D: Dimension,
          S: Storage,
          S2: Storage<Elem = S::Elem>,
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
          S: Storage,
          S::Elem: Eq,
{ }

impl<A> FromIterator<A> for Array<A, Ix>
{
    fn from_iter<I: IntoIterator<Item=A>>(iterable: I) -> Array<A, Ix>
    {
        Array::from_iter(iterable)
    }
}

impl<'a, A, D> IntoIterator for &'a Array<A, D> where
    D: Dimension,
{
    type Item = &'a A;
    type IntoIter = Elements<'a, A, D>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.iter()
    }
}

impl<'a, A, D> IntoIterator for &'a mut Array<A, D> where
    A: Clone,
    D: Dimension,
{
    type Item = &'a mut A;
    type IntoIter = ElementsMut<'a, A, D>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.iter_mut()
    }
}

impl<A: hash::Hash, D: Dimension>
hash::Hash for Array<A, D>
{
    fn hash<S: hash::Hasher>(&self, state: &mut S)
    {
        self.shape().hash(state);
        for elt in self.iter() {
            elt.hash(state)
        }
    }
}

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
