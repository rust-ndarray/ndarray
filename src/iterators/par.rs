

use rayon::par_iter::ParallelIterator;
use rayon::par_iter::IntoParallelIterator;
use rayon::par_iter::IndexedParallelIterator;
use rayon::par_iter::ExactParallelIterator;
use rayon::par_iter::BoundedParallelIterator;
use rayon::par_iter::internal::{Consumer, UnindexedConsumer};
use rayon::par_iter::internal::bridge;
use rayon::par_iter::internal::bridge_unindexed;
use rayon::par_iter::internal::ProducerCallback;
use rayon::par_iter::internal::Producer;
use rayon::par_iter::internal::UnindexedProducer;
use rayon::par_iter::internal::Folder;

use super::AxisIter;
use super::AxisIterMut;
use imp_prelude::*;

/// Iterator wrapper for parallelized implementations.
///
/// **Requires crate feature `"rayon"`**
#[derive(Copy, Clone, Debug)]
pub struct Parallel<I> {
    iter: I,
}

macro_rules! par_iter_wrapper {
    // thread_bounds are either Sync or Send + Sync
    ($iter_name:ident, [$($thread_bounds:tt)*]) => {
    /// This iterator can be turned into a parallel iterator (rayon crate).
    ///
    /// **Requires crate feature `"rayon"`**
    impl<'a, A, D> IntoParallelIterator for $iter_name<'a, A, D>
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
            callback.callback(self.iter)
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

    // This is the real magic, I guess

    impl<'a, A, D> Producer for $iter_name<'a, A, D>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        fn cost(&mut self, len: usize) -> f64 {
            // FIXME: No idea about what this is
            len as f64
        }

        fn split_at(self, i: usize) -> (Self, Self) {
            self.split_at(i)
        }
    }
    }
}

par_iter_wrapper!(AxisIter, [Sync]);
par_iter_wrapper!(AxisIterMut, [Send + Sync]);

use impl_views::ArrayViewPrivate;

macro_rules! par_iter_view_wrapper {
    // thread_bounds are either Sync or Send + Sync
    ($view_name:ident, [$($thread_bounds:tt)*]) => {
    impl<'a, A, D> IntoParallelIterator for $view_name<'a, A, D>
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
            bridge_unindexed(self.iter, consumer)
        }

        fn opt_len(&mut self) -> Option<usize> {
            Some(self.iter.len())
        }
    }

    impl<'a, A, D> UnindexedProducer for $view_name<'a, A, D>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        fn can_split(&self) -> bool {
            self.len() > 1
        }

        fn split(self) -> (Self, Self) {
            let max_axis = self.max_stride_axis();
            let mid = self.len_of(max_axis) / 2;
            let (a, b) = self.split_at(max_axis, mid);
            //println!("Split along axis {:?} at {}\nshapes {:?}, {:?}", max_axis, mid, a.shape(), b.shape());
            (a, b)
        }

        #[cfg(rayon_fold_with)]
        fn fold_with<F>(self, folder: F) -> F
            where F: Folder<Self::Item>,
        {
            self.into_fold(folder, move |f, elt| f.consume(elt))
        }
    }

    }
}

use super::Iter;

par_iter_view_wrapper!(ArrayView, [Sync]);
par_iter_view_wrapper!(ArrayViewMut, [Sync + Send]);
