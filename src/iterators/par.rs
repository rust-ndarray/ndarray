

use rayon::par_iter::ParallelIterator;
use rayon::par_iter::IntoParallelIterator;
use rayon::par_iter::IndexedParallelIterator;
use rayon::par_iter::ExactParallelIterator;
use rayon::par_iter::BoundedParallelIterator;
use rayon::par_iter::internal::{Consumer, UnindexedConsumer};
use rayon::par_iter::internal::bridge;
use rayon::par_iter::internal::ProducerCallback;
use rayon::par_iter::internal::Producer;

use super::AxisIter;
use imp_prelude::*;

/// Parallel iterator wrapper.
pub struct Parallel<I> {
    pub iter: I,
}

impl<I> From<I> for Parallel<I::IntoIter>
    where I: IntoIterator,
{
    fn from(iter: I) -> Self {
        Parallel {
            iter: iter.into_iter()
        }
    }
}

impl<'a, A, D> IntoParallelIterator for AxisIter<'a, A, D>
    where D: Dimension,
          A: Sync,
{
    type Item = <Self as Iterator>::Item;
    type Iter = Parallel<Self>;
    fn into_par_iter(self) -> Self::Iter {
        Parallel {
            iter: self,
        }
    }
}


impl<'a, A, D> ParallelIterator for Parallel<AxisIter<'a, A, D>>
    where D: Dimension,
          A: Sync,
{
    type Item = <AxisIter<'a, A, D> as Iterator>::Item;
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where C: UnindexedConsumer<Self::Item>
    {
        bridge(self, consumer)
    }
}

impl<'a, A, D> IndexedParallelIterator for Parallel<AxisIter<'a, A, D>>
    where D: Dimension,
          A: Sync,
{
    fn with_producer<Cb>(self, callback: Cb) -> Cb::Output
        where Cb: ProducerCallback<Self::Item>
    {
        callback.callback(self.iter)
    }
}

impl<'a, A, D> ExactParallelIterator for Parallel<AxisIter<'a, A, D>>
    where D: Dimension,
          A: Sync,
{
    fn len(&mut self) -> usize {
        self.iter.len()
    }
}

impl<'a, A, D> BoundedParallelIterator for Parallel<AxisIter<'a, A, D>>
    where D: Dimension,
          A: Sync,
{
    fn upper_bound(&mut self) -> usize {
        ExactParallelIterator::len(self)
    }

    fn drive<C>(self, consumer: C) -> C::Result
        where C: Consumer<Self::Item>
    {
        bridge(self, consumer)
    }
}

// This is the real magic, I guess

impl<'a, A, D> Producer for AxisIter<'a, A, D>
    where D: Dimension,
          A: Sync,
{
    fn cost(&mut self, len: usize) -> f64 {
        // FIXME: No idea about what this is
        len as f64
    }

    fn split_at(self, i: usize) -> (Self, Self) {
        self.split_at(i)
    }
}
