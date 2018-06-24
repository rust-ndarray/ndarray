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
            self[0].clone()
        } else {
            let mut rng = thread_rng();
            let pivot_index = rng.gen_range(0, n);
            let partition_index = self.partition_mut(pivot_index);
            if i < partition_index {
                self.slice_mut(s![..partition_index]).ith_mut(i)
            } else if i == partition_index {
                self[i].clone()
            } else {
                self.slice_mut(s![partition_index+1..]).ith_mut(i - (partition_index+1))
            }
        }
    }

    /// Return the index of `self[partition_index]` if `self` were to be sorted
    /// in increasing order.
    ///
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
    /// The method uses Hoare's partition algorithm.
    /// Complexity: O(`n`), where `n` is the number of elements in the array.
    /// Average number of element swaps: n/6 - 1/3 (see
    /// (link)[https://cs.stackexchange.com/questions/11458/quicksort-partitioning-hoare-vs-lomuto/11550])
    ///
    /// **Panics** if `partition_index` is greater than or equal to `n`.
    pub fn partition_mut(&mut self, pivot_index: usize) -> usize
        where A: Ord + Clone,
              S: DataMut
    {
        let pivot_value = self[pivot_index].clone();
        self.swap(pivot_index, 0);

        let n = self.len();
        let mut i = 1;
        let mut j = n - 1;
        loop {
            loop {
                if i > j { break }
                if self[i] >= pivot_value { break }
                i += 1;
            }
            while pivot_value <= self[j] {
                j -= 1;
            }
            if i >= j {
                break
            } else {
                self.swap(i, j);
                i += 1;
                j -= 1;
            }
        }
        self.swap(0, i-1);
        i-1
    }
}

#[test]
fn test_partition_mut() {
    let mut l = vec!(
        arr1(&[1, 3, 2, 10, 10]),
        arr1(&[355, 453, 452, 391, 289, 343,  44, 154, 271,  44, 314, 276, 160,
               469, 191, 138, 163, 308, 395,   3, 416, 391, 210, 354, 200]),
        arr1(&[ 84, 192, 216, 159,  89, 296,  35, 213, 456, 278,  98,  52, 308,
               418, 329, 173, 286, 106, 366, 129, 125, 450,  23, 463, 151]),
    );

    for a in l.iter_mut() {
        let n = a.len();
        let pivot_index = n-1;
        let pivot_value = a[pivot_index].clone();
        let partition_index = a.partition_mut(pivot_index);
        for i in 0..partition_index+1 {
            assert!(a[i] <= pivot_value);
        }
        for j in (partition_index+1)..n {
            assert!(pivot_value <= a[j]);
        }
    }
}
