
use rayon::par_iter::ParallelIterator;
use rayon::par_iter::IndexedParallelIterator;
use rayon::par_iter::ExactParallelIterator;
use rayon::par_iter::BoundedParallelIterator;
use rayon::par_iter::internal::{Consumer, UnindexedConsumer};
use rayon::par_iter::internal::bridge;
use rayon::par_iter::internal::ProducerCallback;
use rayon::par_iter::internal::Producer;
use rayon::par_iter::internal::UnindexedProducer;
use rayon::par_iter::internal::bridge_unindexed;

use ndarray::AxisIter;
use ndarray::AxisIterMut;
use ndarray::{Dimension};
use ndarray::{ArrayView, ArrayViewMut};

use super::NdarrayIntoParallelIterator;

/// Parallel iterator wrapper.
#[derive(Copy, Clone, Debug)]
pub struct Parallel<I> {
    iter: I,
}

/// Parallel producer wrapper.
#[derive(Copy, Clone, Debug)]
struct ParallelProducer<I>(I);

macro_rules! par_iter_wrapper {
    // thread_bounds are either Sync or Send + Sync
    ($iter_name:ident, [$($thread_bounds:tt)*]) => {
    impl<'a, A, D> NdarrayIntoParallelIterator for $iter_name<'a, A, D>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        type Item = <Self as Iterator>::Item;
        type Iter = Parallel<Self>;
        fn into_par_iter(self) -> Self::Iter {
            Parallel {
                iter: self,
            }
        }
    }

    impl<'a, A, D> ParallelIterator for Parallel<$iter_name<'a, A, D>>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        type Item = <$iter_name<'a, A, D> as Iterator>::Item;
        fn drive_unindexed<C>(self, consumer: C) -> C::Result
            where C: UnindexedConsumer<Self::Item>
        {
            bridge(self, consumer)
        }

        fn opt_len(&mut self) -> Option<usize> {
            Some(self.iter.len())
        }
    }

    impl<'a, A, D> IndexedParallelIterator for Parallel<$iter_name<'a, A, D>>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        fn with_producer<Cb>(self, callback: Cb) -> Cb::Output
            where Cb: ProducerCallback<Self::Item>
        {
            callback.callback(ParallelProducer(self.iter))
        }
    }

    impl<'a, A, D> ExactParallelIterator for Parallel<$iter_name<'a, A, D>>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        fn len(&mut self) -> usize {
            ExactSizeIterator::len(&self.iter)
        }
    }

    impl<'a, A, D> BoundedParallelIterator for Parallel<$iter_name<'a, A, D>>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        fn upper_bound(&mut self) -> usize {
            ExactSizeIterator::len(&self.iter)
        }

        fn drive<C>(self, consumer: C) -> C::Result
            where C: Consumer<Self::Item>
        {
            bridge(self, consumer)
        }
    }

    impl<'a, A, D> IntoIterator for ParallelProducer<$iter_name<'a, A, D>>
        where D: Dimension,
    {
        type IntoIter = $iter_name<'a, A, D>;
        type Item = <Self::IntoIter as Iterator>::Item;

        fn into_iter(self) -> Self::IntoIter {
            self.0
        }
    }

    // This is the real magic, I guess
    impl<'a, A, D> Producer for ParallelProducer<$iter_name<'a, A, D>>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        fn cost(&mut self, len: usize) -> f64 {
            // FIXME: No idea about what this is
            len as f64
        }

        fn split_at(self, i: usize) -> (Self, Self) {
            let (a, b) = self.0.split_at(i);
            (ParallelProducer(a), ParallelProducer(b))
        }
    }

    }
}


par_iter_wrapper!(AxisIter, [Sync]);
par_iter_wrapper!(AxisIterMut, [Send + Sync]);



macro_rules! par_iter_view_wrapper {
    // thread_bounds are either Sync or Send + Sync
    ($view_name:ident, [$($thread_bounds:tt)*]) => {
    impl<'a, A, D> NdarrayIntoParallelIterator for $view_name<'a, A, D>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        type Item = <Self as IntoIterator>::Item;
        type Iter = Parallel<Self>;
        fn into_par_iter(self) -> Self::Iter {
            Parallel {
                iter: self,
            }
        }
    }


    impl<'a, A, D> ParallelIterator for Parallel<$view_name<'a, A, D>>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        type Item = <$view_name<'a, A, D> as IntoIterator>::Item;
        fn drive_unindexed<C>(self, consumer: C) -> C::Result
            where C: UnindexedConsumer<Self::Item>
        {
            bridge_unindexed(ParallelProducer(self.iter), consumer)
        }

        fn opt_len(&mut self) -> Option<usize> {
            Some(self.iter.len())
        }
    }

    impl<'a, A, D> UnindexedProducer for ParallelProducer<$view_name<'a, A, D>>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        fn can_split(&self) -> bool {
            self.0.len() > 1
        }

        fn split(self) -> (Self, Self) {
            let array = self.0;
            let max_axis = array.max_stride_axis();
            let mid = array.len_of(max_axis) / 2;
            let (a, b) = array.split_at(max_axis, mid);
            (ParallelProducer(a), ParallelProducer(b))
        }

        #[cfg(rayon_fold_with)]
        fn fold_with<F>(self, folder: F) -> F
            where F: Folder<Self::Item>,
        {
            self.into_iter().fold(folder, move |f, elt| f.consume(elt))
        }
    }

    impl<'a, A, D> IntoIterator for ParallelProducer<$view_name<'a, A, D>>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        type Item = <$view_name<'a, A, D> as IntoIterator>::Item;
        type IntoIter = <$view_name<'a, A, D> as IntoIterator>::IntoIter;
        fn into_iter(self) -> Self::IntoIter {
            self.0.into_iter()
        }
    }

    }
}

par_iter_view_wrapper!(ArrayView, [Sync]);
par_iter_view_wrapper!(ArrayViewMut, [Sync + Send]);
