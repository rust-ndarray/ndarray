// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use imp_prelude::*;
use error::{ShapeError, ErrorKind, from_kind};

/// Stack arrays along the given axis.
///
/// ***Errors*** if the arrays have mismatching shapes, apart from along `axis`.
/// (may be made more flexible in the future).<br>
/// ***Errors*** if `arrays` is empty, if `axis` is out of bounds,
/// if the result is larger than is possible to represent.
///
/// ```
/// use ndarray::{arr2, Axis, stack};
///
/// let a = arr2(&[[2., 2.],
///                [3., 3.]]);
/// assert!(
///     stack(Axis(0), &[a.view(), a.view()])
///     == Ok(arr2(&[[2., 2.],
///                  [3., 3.],
///                  [2., 2.],
///                  [3., 3.]]))
/// );
/// ```
pub fn stack<'a, A, D>(axis: Axis, arrays: &[ArrayView<'a, A, D>])
    -> Result<Array<A, D>, ShapeError>
    where A: Copy,
          D: RemoveAxis
{
    if arrays.len() == 0 {
        return Err(from_kind(ErrorKind::Unsupported));
    }
    let mut res_dim = arrays[0].raw_dim();
    if axis.index() >= res_dim.ndim() {
        return Err(from_kind(ErrorKind::OutOfBounds));
    }
    let common_dim = res_dim.remove_axis(axis);
    if arrays.iter().any(|a| a.raw_dim().remove_axis(axis) != common_dim) {
        return Err(from_kind(ErrorKind::IncompatibleShape));
    }

    let stacked_dim = arrays.iter()
                            .fold(0, |acc, a| acc + a.shape().axis(axis));
    res_dim.set_axis(axis, stacked_dim);

    // we can safely use uninitialized values here because they are Copy
    // and we will only ever write to them
    let size = res_dim.size();
    let mut v = Vec::with_capacity(size);
    unsafe {
        v.set_len(size);
    }
    let mut res = try!(Array::from_shape_vec(res_dim, v));

    {
        let mut assign_view = res.view_mut();
        for array in arrays {
            let len = array.shape().axis(axis);
            let (mut front, rest) = assign_view.split_at(axis, len);
            front.assign(array);
            assign_view = rest;
        }
    }
    Ok(res)
}

/// Stack arrays along the given axis.
///
/// Uses the [`stack`][1] function, calling `ArrayView::from(&a)` on each
/// argument `a`.
///
/// [1]: fn.stack.html
///
/// ***Panics*** if the `stack` function would return an error.
///
/// ```
/// #[macro_use(stack)]
/// extern crate ndarray;
///
/// use ndarray::{arr2, Axis, stack};
///
/// # fn main() {
///
/// let a = arr2(&[[2., 2.],
///                [3., 3.]]);
/// assert!(
///     stack![Axis(0), a, a]
///     == arr2(&[[2., 2.],
///               [3., 3.],
///               [2., 2.],
///               [3., 3.]])
/// );
/// # }
/// ```
#[macro_export]
macro_rules! stack {
    ($axis:expr, $( $array:expr ),+ ) => {
        $crate::stack($axis, &[ $($crate::ArrayView::from(&$array) ),* ]).unwrap()
    }
}
