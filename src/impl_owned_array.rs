
use std::rc::Rc;

use imp_prelude::*;
use {
    OwnedRepr,
    OwnedRcRepr,
};

impl<A, D> ArrayBase<OwnedRepr<A>, D>
    where D: Dimension
{
    /// Return a vector of the elements in the array, in the way they are
    /// stored internally.
    ///
    /// If the array is in standard memory layout, the logical element order
    /// of the array (`.iter()` order) and of the returned vector will be the same.
    pub fn into_raw_vec(self) -> Vec<A> {
        self.data.0
    }
}

// RcArray
impl<A, D> ArrayBase<OwnedRcRepr<A>, D>
    where A: Clone,
          D: Dimension
{
    /// Convert an `RcArray` into `Array`; cloning the array elements to unshare
    /// them if necessary.
    pub fn into_owned(mut self) -> Array<A, D> {
        <_>::ensure_unique(&mut self);
        let data = OwnedRepr(Rc::try_unwrap(self.data.0).ok().unwrap());
        ArrayBase {
            data: data,
            ptr: self.ptr,
            dim: self.dim,
            strides: self.strides,
        }
    }
}
