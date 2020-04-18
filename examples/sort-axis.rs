use ndarray::prelude::*;
use ndarray::{Data, RemoveAxis, Zip};

use std::cmp::Ordering;
use std::ptr::copy_nonoverlapping;

// Type invariant: Each index appears exactly once
#[derive(Clone, Debug)]
pub struct Permutation {
    indices: Vec<usize>,
}

impl Permutation {
    /// Checks if the permutation is correct
    pub fn from_indices(v: Vec<usize>) -> Result<Self, ()> {
        let perm = Permutation { indices: v };
        if perm.correct() {
            Ok(perm)
        } else {
            Err(())
        }
    }

    fn correct(&self) -> bool {
        let axis_len = self.indices.len();
        let mut seen = vec![false; axis_len];
        for &i in &self.indices {
            match seen.get_mut(i) {
                None => return false,
                Some(s) => {
                    if *s {
                        return false;
                    } else {
                        *s = true;
                    }
                }
            }
        }
        true
    }
}

pub trait SortArray {
    /// ***Panics*** if `axis` is out of bounds.
    fn identity(&self, axis: Axis) -> Permutation;
    fn sort_axis_by<F>(&self, axis: Axis, less_than: F) -> Permutation
    where
        F: FnMut(usize, usize) -> bool;
}

pub trait PermuteArray {
    type Elem;
    type Dim;
    fn permute_axis(self, axis: Axis, perm: &Permutation) -> Array<Self::Elem, Self::Dim>
    where
        Self::Elem: Clone,
        Self::Dim: RemoveAxis;
}

impl<A, S, D> SortArray for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn identity(&self, axis: Axis) -> Permutation {
        Permutation {
            indices: (0..self.len_of(axis)).collect(),
        }
    }

    fn sort_axis_by<F>(&self, axis: Axis, mut less_than: F) -> Permutation
    where
        F: FnMut(usize, usize) -> bool,
    {
        let mut perm = self.identity(axis);
        perm.indices.sort_by(move |&a, &b| {
            if less_than(a, b) {
                Ordering::Less
            } else if less_than(b, a) {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
        perm
    }
}

impl<A, D> PermuteArray for Array<A, D>
where
    D: Dimension,
{
    type Elem = A;
    type Dim = D;

    fn permute_axis(self, axis: Axis, perm: &Permutation) -> Array<A, D>
    where
        D: RemoveAxis,
    {
        let axis = axis;
        let axis_len = self.len_of(axis);
        assert_eq!(axis_len, perm.indices.len());
        debug_assert!(perm.correct());

        let mut result = Array::maybe_uninit(self.dim());

        // panic-critical begin: we must not panic
        unsafe {
            // logically move ownership of all elements from self into result
            // the result realizes this ownership at .assume_init() further down
            let mut moved_elements = 0;
            for i in 0..axis_len {
                let perm_i = perm.indices[i];
                Zip::from(result.index_axis_mut(axis, perm_i))
                    .and(self.index_axis(axis, i))
                    .apply(|to, from| {
                        copy_nonoverlapping(from, to.as_mut_ptr(), 1);
                        moved_elements += 1;
                    });
            }
            // forget moved array elements but not its vec
            // old_storage drops empty
            let mut old_storage = self.into_raw_vec();
            old_storage.set_len(0);

            debug_assert_eq!(result.len(), moved_elements);
            result.assume_init()
        }
        // panic-critical end
    }
}

fn main() {
    let a = Array::linspace(0., 63., 64).into_shape((8, 8)).unwrap();
    let strings = a.map(|x| x.to_string());

    let perm = a.sort_axis_by(Axis(1), |i, j| a[[i, 0]] > a[[j, 0]]);
    println!("{:?}", perm);
    let b = a.permute_axis(Axis(0), &perm);
    println!("{:?}", b);

    println!("{:?}", strings);
    let c = strings.permute_axis(Axis(1), &perm);
    println!("{:?}", c);
}
