use ndarray::{Array, RcArray, Dimension, ArrayView, ArrayViewMut};

use NdarrayIntoParallelIterator as IntoParallelIterator;
use Parallel;

impl<'a, A, D> IntoParallelIterator for &'a Array<A, D>
    where D: Dimension,
          A: Sync
{
    type Item = &'a A;
    type Iter = Parallel<ArrayView<'a, A, D>>;
    fn into_par_iter(self) -> Self::Iter {
        self.view().into_par_iter()
    }
}

impl<'a, A, D> IntoParallelIterator for &'a RcArray<A, D>
    where D: Dimension,
          A: Sync
{
    type Item = &'a A;
    type Iter = Parallel<ArrayView<'a, A, D>>;
    fn into_par_iter(self) -> Self::Iter {
        self.view().into_par_iter()
    }
}

impl<'a, A, D> IntoParallelIterator for &'a mut Array<A, D>
    where D: Dimension,
          A: Sync + Send
{
    type Item = &'a mut A;
    type Iter = Parallel<ArrayViewMut<'a, A, D>>;
    fn into_par_iter(self) -> Self::Iter {
        self.view_mut().into_par_iter()
    }
}

impl<'a, A, D> IntoParallelIterator for &'a mut RcArray<A, D>
    where D: Dimension,
          A: Sync + Send + Clone,
{
    type Item = &'a mut A;
    type Iter = Parallel<ArrayViewMut<'a, A, D>>;
    fn into_par_iter(self) -> Self::Iter {
        self.view_mut().into_par_iter()
    }
}
