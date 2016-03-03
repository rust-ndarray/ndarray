
use imp_prelude::*;
use libnum;

/// A slice extension trait for concatenating arrays.
pub trait ArrayStackingExt {
    type Output;

    /// Stack the given arrays along the specified axis.
    ///
    /// *Panics* if axis is out of bounds
    /// *Panics* if the slice is empty.
    fn stack(&self, axis: Axis) -> Self::Output;
}

impl<'a, A, D> ArrayStackingExt for [ArrayView<'a, A, D>]
    where A: Clone + libnum::Zero,
          D: Dimension + RemoveAxis
{
    type Output = OwnedArray<A, D>;

    fn stack(&self, axis: Axis) -> <Self as ArrayStackingExt>::Output {
        assert!(self.len() > 0);
        let mut res_dim = self[0].dim().clone();
        let stacked_dim = self.iter()
                              .fold(0, |acc, a| acc + a.dim().index(axis));
        *res_dim.index_mut(axis) = stacked_dim;
        let mut res = OwnedArray::zeros(res_dim);

        let mut array_iter = self.iter();
        let mut in_iter = array_iter.next().unwrap().axis_iter(axis);
        let mut cum = *self[0].dim().index(axis);
        for (ind, mut out) in res.axis_iter_mut(axis).enumerate() {
            if ind == cum {
                let cur_array = array_iter.next().unwrap();
                cum += *cur_array.dim().index(axis);
                in_iter = cur_array.axis_iter(axis);
            }
            out.assign(&in_iter.next().unwrap());
        }
        res
    }
}
