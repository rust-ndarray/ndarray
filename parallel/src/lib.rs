
extern crate ndarray;
extern crate rayon;

use rayon::par_iter::ParallelIterator;

pub trait NdarrayIntoParallelIterator {
    type Iter: ParallelIterator<Item=Self::Item>;
    type Item: Send;
    fn into_par_iter(self) -> Self::Iter;
}

pub use par::Parallel;

mod par;
