
use imp_prelude::*;
use error::{ShapeError, ErrorKind, from_kind};

/// Stack arrays along the given axis.
pub fn stack<'a, A, D>(axis: Axis, arrays: &[ArrayView<'a, A, D>])
    -> Result<OwnedArray<A, D>, ShapeError>
    where A: Copy,
          D: Dimension + RemoveAxis
{
    if arrays.len() == 0 {
        return Err(from_kind(ErrorKind::Unsupported));
    }
    let mut res_dim = arrays[0].dim();
    if axis.axis() >= res_dim.ndim() {
        return Err(from_kind(ErrorKind::OutOfBounds));
    }
    let common_dim = res_dim.remove_axis(axis);
    if arrays.iter().any(|a| a.dim().remove_axis(axis) != common_dim) {
        return Err(from_kind(ErrorKind::IncompatibleShape));
    }

    let stacked_dim = arrays.iter()
                            .fold(0, |acc, a| acc + a.dim().index(axis));
    *res_dim.index_mut(axis) = stacked_dim;

    // we can safely use uninitialized values here because they are Copy
    // and we will only ever write to them
    let size = res_dim.size();
    let mut v = Vec::with_capacity(size);
    unsafe {
        v.set_len(size);
    }
    let mut res = try!(OwnedArray::from_vec_dim(res_dim, v));

    {
        let mut assign_view = res.view_mut();
        for array in arrays {
            let len = *array.dim().index(axis);
            let (mut front, rest) = assign_view.split_at(axis, len);
            front.assign(array);
            assign_view = rest;
        }
    }
    Ok(res)
}

#[macro_export]
macro_rules! stack {
    ($axis:expr, $( $a:expr ),+ ) => {
        ndarray::stack($axis, &[ $($a.view() ),* ]).unwrap()
    }
}
