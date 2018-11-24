
use rayon::iter::ParallelIterator;

pub trait NdarrayIntoParallelIterator {
    type Iter: ParallelIterator<Item=Self::Item>;
    type Item: Send;
    fn into_par_iter(self) -> Self::Iter;
}

pub trait NdarrayIntoParallelRefIterator<'x> {
    type Iter: ParallelIterator<Item=Self::Item>;
    type Item: Send + 'x;
    fn par_iter(&'x self) -> Self::Iter;
}

pub trait NdarrayIntoParallelRefMutIterator<'x> {
    type Iter: ParallelIterator<Item=Self::Item>;
    type Item: Send + 'x;
    fn par_iter_mut(&'x mut self) -> Self::Iter;
}

impl<'data, I: 'data + ?Sized> NdarrayIntoParallelRefIterator<'data> for I
    where &'data I: NdarrayIntoParallelIterator
{
    type Iter = <&'data I as NdarrayIntoParallelIterator>::Iter;
    type Item = <&'data I as NdarrayIntoParallelIterator>::Item;

    fn par_iter(&'data self) -> Self::Iter {
        self.into_par_iter()
    }
}

impl<'data, I: 'data + ?Sized> NdarrayIntoParallelRefMutIterator<'data> for I
    where &'data mut I: NdarrayIntoParallelIterator
{
    type Iter = <&'data mut I as NdarrayIntoParallelIterator>::Iter;
    type Item = <&'data mut I as NdarrayIntoParallelIterator>::Item;

    fn par_iter_mut(&'data mut self) -> Self::Iter {
        self.into_par_iter()
    }
}
