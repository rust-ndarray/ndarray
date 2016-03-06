// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
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

