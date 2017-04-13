
use rayon::iter::ParallelIterator;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ExactParallelIterator;
use rayon::iter::BoundedParallelIterator;
use rayon::iter::internal::{Consumer, UnindexedConsumer};
use rayon::iter::internal::bridge;
use rayon::iter::internal::ProducerCallback;
use rayon::iter::internal::Producer;
use rayon::iter::internal::UnindexedProducer;
use rayon::iter::internal::bridge_unindexed;
use rayon::iter::internal::Folder;

use ndarray::iter::AxisIter;
use ndarray::iter::AxisIterMut;
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
        type IntoIter = $iter_name<'a, A, D>;
        type Item = <Self::IntoIter as Iterator>::Item;

        fn into_iter(self) -> Self::IntoIter {
            self.0
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
            None
        }
    }

    impl<'a, A, D> UnindexedProducer for ParallelProducer<$view_name<'a, A, D>>
        where D: Dimension,
              A: $($thread_bounds)*,
    {
        type Item = <$view_name<'a, A, D> as IntoIterator>::Item;
        fn split(self) -> (Self, Option<Self>) {
            if self.0.len() <= 1 {
                return (self, None)
            }
            let array = self.0;
            let max_axis = array.max_stride_axis();
            let mid = array.len_of(max_axis) / 2;
            let (a, b) = array.split_at(max_axis, mid);
            (ParallelProducer(a), Some(ParallelProducer(b)))
        }

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


use ndarray::{Zip, NdProducer, FoldWhile};

macro_rules! zip_impl {
    ($([$($p:ident)*],)+) => {
        $(
        #[allow(non_snake_case)]
        impl<Dim: Dimension, $($p: NdProducer<Dim=Dim>),*> NdarrayIntoParallelIterator for Zip<($($p,)*), Dim>
            where $($p::Item : Send , )*
                  $($p : Send , )*
        {
            type Item = ($($p::Item ,)*);
            type Iter = Parallel<Self>;
            fn into_par_iter(self) -> Self::Iter {
                Parallel {
                    iter: self,
                }
            }
        }

        #[allow(non_snake_case)]
        impl<Dim: Dimension, $($p: NdProducer<Dim=Dim>),*> ParallelIterator for Parallel<Zip<($($p,)*), Dim>>
            where $($p::Item : Send , )*
                  $($p : Send , )*
        {
            type Item = ($($p::Item ,)*);

            fn drive_unindexed<Cons>(self, consumer: Cons) -> Cons::Result
                where Cons: UnindexedConsumer<Self::Item>
            {
                bridge_unindexed(ParallelProducer(self.iter), consumer)
            }

            fn opt_len(&mut self) -> Option<usize> {
                None
            }
        }

        #[allow(non_snake_case)]
        impl<Dim: Dimension, $($p: NdProducer<Dim=Dim>),*> UnindexedProducer for ParallelProducer<Zip<($($p,)*), Dim>>
            where $($p : Send , )*
                  $($p::Item : Send , )*
        {
            type Item = ($($p::Item ,)*);

            fn split(self) -> (Self, Option<Self>) {
                if self.0.size() <= 1 {
                    return (self, None)
                }
                let (a, b) = self.0.split();
                (ParallelProducer(a), Some(ParallelProducer(b)))
            }

            fn fold_with<Fold>(self, folder: Fold) -> Fold
                where Fold: Folder<Self::Item>,
            {
                self.0.fold_while(folder, |mut folder, $($p),*| {
                    folder = folder.consume(($($p ,)*));
                    if folder.full() {
                        FoldWhile::Done(folder)
                    } else {
                        FoldWhile::Continue(folder)
                    }
                }).into_inner()
            }
        }
        )+
    }
}

zip_impl!{
    [P1],
    [P1 P2],
    [P1 P2 P3],
    [P1 P2 P3 P4],
    [P1 P2 P3 P4 P5],
    [P1 P2 P3 P4 P5 P6],
}
