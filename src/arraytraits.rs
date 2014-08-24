use std::hash;

use super::{Array, Dimension, Ix};

impl<A: PartialEq, D: Dimension>
PartialEq for Array<A, D>
{
    /// Return `true` if all elements of `self` and `other` are equal.
    ///
    /// **Fail** if shapes are not equal.
    fn eq(&self, other: &Array<A, D>) -> bool
    {
        assert!(self.shape() == other.shape());
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
