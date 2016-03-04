
use imp_prelude::*;

pub fn stack<'a, A, D>(arrays: &[ArrayView<'a, A, D>], axis: Axis)
    -> OwnedArray<A, D>
    where A: Copy,
          D: Dimension + RemoveAxis
{
    assert!(arrays.len() > 0);
    let mut res_dim = arrays[0].dim().clone();
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
    let mut res = OwnedArray::from_vec_dim(res_dim, v).unwrap();

    {
        let mut assign_view = res.view_mut();
        for array in arrays {
            let len = *array.dim().index(axis);
            let (mut front, rest) = assign_view.split_at(axis, len);
            front.assign(array);
            assign_view = rest;
        }
    }
    res
}
