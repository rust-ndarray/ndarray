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
