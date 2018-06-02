// Copyright 2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//! Methods for one-dimensional arrays.
use imp_prelude::*;

use rand::prelude::*;
use rand::thread_rng;

impl<A, S> ArrayBase<S, Ix1>
    where S: Data<Elem=A>,
{
    /// Return an vector with the elements of the one-dimensional array.
    pub fn to_vec(&self) -> Vec<A>
        where A: Clone,
    {
        if let Some(slc) = self.as_slice() {
            slc.to_vec()
        } else {
            ::iterators::to_vec(self.iter().map(|x| x.clone()))
        }
    }

    /// Return the element that would occupy the `i`-th position if the array
    /// were sorted in increasing order.
    ///
    /// The array is shuffled **in place** to retrieve the desired element:
    /// no copy of the array is allocated.
    /// No assumptions should be made on the ordering of elements
    /// after this computation.
    ///
    /// Complexity ([quickselect](https://en.wikipedia.org/wiki/Quickselect)):
    /// - average case: O(`n`);
    /// - worst case: O(`n`^2);
    /// where n is the number of elements in the array.
    ///
    /// **Panics** if `i` is greater than or equal to `n`.
    pub fn ith_mut(&mut self, i: usize) -> A
        where A: Ord + Clone,
              S: DataMut,
    {
        let n = self.len();
        if n == 1 {
            (&self[0]).clone()
        } else {
            let pivot_index = random_pivot(n);
            let partition_index = self.view_mut().partition_mut(pivot_index);
            if i == partition_index {
                (&self[partition_index]).clone()
            } else if i < partition_index {
                self.slice_mut(s![0..partition_index]).ith_mut(i)
            } else {
                self.slice_mut(s![(partition_index+1)..n]).ith_mut(i - partition_index - 1)
            }
        }
    }

    /// Return the index of `self[partition_index]` if `self` were to be sorted
    /// in increasing order.
    /// `self` elements are rearranged in such a way that `self[partition_index]`
    /// is in the position it would be in an array sorted in increasing order.
    /// All elements smaller than `self[partition_index]` are moved to its
    /// left and all elements equal or greater than `self[partition_index]`
    /// are moved to its right.
    /// The ordering of the elements in the two partitions is undefined.
    ///
    /// `self` is shuffled **in place** to operate the desired partition:
    /// no copy of the array is allocated.
    ///
    /// Complexity: O(`n`), where `n` is the number of elements in the array.
    ///
    /// **Panics** if `partition_index` is greater than or equal to `n`.
    pub fn partition_mut(&mut self, partition_index: usize) -> usize
        where A: Ord + Clone,
              S: DataMut
    {
        let n = self.len();
        let partition_value = (&self[partition_index]).clone();
        self.swap(partition_index, n-1);
        let mut partition_boundary_index = 0;
        for j in 0..n-1 {
            if self[j] < partition_value {
                self.swap(partition_boundary_index, j);
                partition_boundary_index += 1;
            }
        }
        self.swap(partition_boundary_index, n-1);
        partition_boundary_index
    }
}

fn random_pivot(n: usize) -> usize
{
    let mut rng = thread_rng();
    rng.gen_range(0, n)
}
