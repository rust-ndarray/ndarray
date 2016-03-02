
use imp_prelude::*;

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
    where A: Clone,
          D: Dimension
{
    type Output = OwnedArray<A, D>;

    fn stack(&self, Axis(axis): Axis) -> <Self as ArrayStackingExt>::Output {
        assert!(self.len() > 0);
        let mut res_dim = self[0].dim().clone();
        let stacked_dim = self.iter()
                              .fold(0, |acc, a| acc + a.dim().slice()[axis]);
        res_dim.slice_mut()[axis] = stacked_dim;
        let mut res = OwnedArray::zeros(stacked_dim);
        unimplemented!()
    }
}
