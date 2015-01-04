use super::{Dimension, Ix};

/// An iterator of the indexes of an array shape.
///
/// Iterator element type is `D`.
#[derive(Clone)]
pub struct Indexes<D> {
    dim: D,
    index: Option<D>,
}

impl<D: Dimension> Indexes<D>
{
    /// Create an iterator over the array shape `dim`.
    pub fn new(dim: D) -> Indexes<D>
    {
        Indexes {
            index: dim.first_index(),
            dim: dim,
        }
    }
}

/// Like `range`, except with array indexes.
#[inline]
pub fn ixrange(a: Ix, b: Ix) -> Indexes<Ix>
{
    Indexes {
        index: if a >= b { None } else { Some(a) },
        dim: b,
    }
}

impl Indexes<()>
{
    /// Create an iterator over the array shape `a`.
    pub fn new1(a: Ix) -> Indexes<Ix>
    {
        Indexes {
            index: a.first_index(),
            dim: a,
        }
    }

    /// Create an iterator over the array shape `(a, b)`.
    pub fn new2(a: Ix, b: Ix) -> Indexes<(Ix, Ix)>
    {
        Indexes {
            index: (a, b).first_index(),
            dim: (a, b),
        }
    }

    /// Create an iterator over the array shape `(a, b, c)`.
    pub fn new3(a: Ix, b: Ix, c: Ix) -> Indexes<(Ix, Ix, Ix)>
    {
        Indexes {
            index: (a, b, c).first_index(),
            dim: (a, b, c),
        }
    }
}


impl<D: Dimension> Iterator for Indexes<D>
{
    type Item = D;
    #[inline]
    fn next(&mut self) -> Option<D>
    {
        let index = match self.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        self.index = self.dim.next_for(index.clone());
        Some(index)
    }

    fn size_hint(&self) -> (uint, Option<uint>)
    {
        let l = match self.index {
            None => 0,
            Some(ref ix) => {
                let gone = self.dim.default_strides().slice().iter()
                            .zip(ix.slice().iter())
                                 .fold(0u, |s, (&a, &b)| s + a as uint * b as uint);
                self.dim.size() - gone
            }
        };
        (l, Some(l))
    }
}

