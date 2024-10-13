use ndarray::{array, ArrayBase, ArrayRef, Data, LayoutRef, RawData, RawRef};

fn takes_base_raw<S: RawData, D>(arr: &ArrayBase<S, D>)
{
    takes_rawref(arr.as_ref()); // Doesn't work
    takes_layout(arr.as_ref());
}

#[allow(dead_code)]
fn takes_base<S: Data, D>(arr: &ArrayBase<S, D>)
{
    takes_base_raw(arr);
    takes_arrref(arr); // Doesn't work
    takes_rawref(arr); // Doesn't work
    takes_layout(arr);
}

fn takes_arrref<A, D>(_arr: &ArrayRef<A, D>)
{
    takes_rawref(_arr);
    takes_layout(_arr);
}

fn takes_rawref<A, D>(_arr: &RawRef<A, D>)
{
    takes_layout(_arr);
}

fn takes_layout<A, D>(_arr: &LayoutRef<A, D>) {}

#[test]
fn tester()
{
    let arr = array![1, 2, 3];
    takes_base_raw(&arr);
    takes_arrref(&arr);
    takes_rawref(&arr); // Doesn't work
    takes_layout(&arr);
}
