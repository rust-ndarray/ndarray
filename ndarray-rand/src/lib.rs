// Copyright 2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


extern crate rand;
extern crate ndarray;

use std::ptr;

use rand::Rng;
use rand::distributions::Sample;
use rand::distributions::IndependentSample;

use ndarray::{
    ArrayBase,
    Dimension,
    Data,
    DataOwned,
};

/// Constructors for n-dimensional arrays with random elements.
///
/// This trait extends ndarray’s `ArrayBase` and can not be implemented
/// for other types.
///
/// The default Rng is a fast automatically seeded rng (currently `rand::weak_rng`).
pub trait RandomExt<S, D>
    where S: DataOwned,
          D: Dimension,
{
    /// Create an array with shape `dim` with elements drawn from
    /// `distribution`  using the default rng.
    ///
    /// ```
    /// extern crate rand;
    /// extern crate ndarray;
    /// extern crate ndarray_rand;
    ///
    /// use rand::distributions::Range;
    /// use ndarray::OwnedArray;
    /// use ndarray_rand::RandomExt;
    ///
    /// # fn main() {
    /// let a = OwnedArray::random((2, 5), Range::new(0., 10.));
    /// println!("{:8.4}", a);
    /// // Output:
    /// // [[  8.6900,   6.9824,   3.8922,   6.5861,   2.4890],
    /// //  [  0.0914,   5.5186,   5.8135,   5.2361,   3.1879]]
    /// # }
    fn random<IdS>(dim: D, distribution: IdS) -> ArrayBase<S, D>
        where IdS: IndependentSample<S::Elem>;

    /// Create an array with shape `dim` with elements drawn from
    /// `distribution`, using a specific Rng `rng`.
    fn random_using<IdS, R>(dim: D, distribution: IdS, rng: &mut R) -> ArrayBase<S, D>
        where IdS: IndependentSample<S::Elem>,
              R: Rng;
}

impl<S, D> RandomExt<S, D> for ArrayBase<S, D>
    where S: DataOwned,
          D: Dimension,
{
    fn random<IdS>(dim: D, dist: IdS) -> ArrayBase<S, D>
        where IdS: IndependentSample<S::Elem>
    {
        Self::random_using(dim, dist, &mut rand::weak_rng())
    }

    fn random_using<IdS, R>(dim: D, dist: IdS, rng: &mut R) -> ArrayBase<S, D>
        where IdS: IndependentSample<S::Elem>,
              R: Rng
    {
        unsafe {
            let elements = to_vec((0..dim.size()).map(move |_| dist.ind_sample(rng)));
            Self::from_vec_dim(dim, elements).unwrap()
        }
    }
}

/// Like Iterator::collect, but only for trusted length iterators
unsafe fn to_vec<I>(iter: I) -> Vec<I::Item>
    where I: ExactSizeIterator
{
    // Use an `unsafe` block to do this efficiently.
    // We know that iter will produce exactly .size() elements,
    // and the loop can vectorize if it's clean (without branch to grow the vector).
    let (size, _) = iter.size_hint();
    let mut result = Vec::with_capacity(size);
    let mut out_ptr = result.as_mut_ptr();
    let mut len = 0;
    for elt in iter {
        ptr::write(out_ptr, elt);
        len += 1;
        result.set_len(len);
        out_ptr = out_ptr.offset(1);
    }
    debug_assert_eq!(size, result.len());
    result
}
