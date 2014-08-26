use serialize::{Encodable, Encoder, Decodable, Decoder};

use std::hash;

use super::{Array, Dimension, Ix};

impl<'a, A, D: Dimension> Index<D, A> for Array<A, D>
{
    #[inline]
    fn index(&self, index: &D) -> &A {
        self.at(index.clone()).unwrap()
    }
}

impl<'a, A: Clone, D: Dimension> IndexMut<D, A> for Array<A, D>
{
    #[inline]
    fn index_mut(&mut self, index: &D) -> &mut A {
        self.at_mut(index.clone()).unwrap()
    }
}


impl<A: PartialEq, D: Dimension>
PartialEq for Array<A, D>
{
    /// Return `true` if the array shapes and all elements of `self` and
    /// `other` are equal. Return `false` otherwise.
    fn eq(&self, other: &Array<A, D>) -> bool
    {
        self.shape() == other.shape() &&
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<A: Eq, D: Dimension>
Eq for Array<A, D> {}

impl<A> FromIterator<A> for Array<A, Ix>
{
    fn from_iter<I: Iterator<A>>(it: I) -> Array<A, Ix>
    {
        Array::from_iter(it)
    }
}

impl<S: hash::Writer, A: hash::Hash<S>, D: Dimension>
hash::Hash<S> for Array<A, D>
{
    fn hash(&self, state: &mut S)
    {
        self.shape().hash(state);
        for elt in self.iter() {
            elt.hash(state)
        }
    }
}

// Use version number so we can add a packed format later.
static ARRAY_FORMAT_VERSION: u8 = 1u8;

impl<A: Encodable<S, E>, D: Dimension + Encodable<S, E>, E, S: Encoder<E>>
    Encodable<S, E> for Array<A, D>
{
    fn encode(&self, s: &mut S) -> Result<(), E>
    {
        s.emit_struct("Array", 0, |e| {
            try!(e.emit_struct_field("v", 0, |e| {
                ARRAY_FORMAT_VERSION.encode(e)
            }))
            // FIXME: Write self.dim as a slice (self.shape)
            // The problem is decoding it.
            try!(e.emit_struct_field("dim", 1,
                                           |e| self.dim.encode(e)))
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
            }))
            Ok(())
        })
    }
}

impl<A: Decodable<S, E>, D: Dimension + Decodable<S, E>, E, S: Decoder<E>>
    Decodable<S, E> for Array<A, D>
{
    fn decode(d: &mut S) -> Result<Array<A, D>, E>
    {
        d.read_struct("Array", 0, |d| {
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
                            for i in range(0, len) {
                                elements.push(try!(d.read_seq_elt::<A>(i, Decodable::decode)))
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
