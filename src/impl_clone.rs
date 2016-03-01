use imp_prelude::*;
use DataClone;

impl<S: DataClone, D: Clone> Clone for ArrayBase<S, D> {
    fn clone(&self) -> ArrayBase<S, D> {
        unsafe {
            let (data, ptr) = self.data.clone_with_ptr(self.ptr);
            ArrayBase {
                data: data,
                ptr: ptr,
                dim: self.dim.clone(),
                strides: self.strides.clone(),
            }
        }
    }
}

impl<S: DataClone + Copy, D: Copy> Copy for ArrayBase<S, D> {}

