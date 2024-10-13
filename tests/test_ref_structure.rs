use ndarray::{array, ArrayBase, ArrayRef, Data, LayoutRef, RawData, RawRef};

fn takes_base_raw<S: RawData, D>(arr: &ArrayBase<S, D>)
{
    takes_rawref(arr.as_ref()); // Doesn't work
    takes_layout(arr.as_ref());
}

fn takes_base<S: Data, D>(arr: &ArrayBase<S, D>)
{
    takes_base_raw(arr);
    takes_arrref(arr); // Doesn't work
    takes_rawref(arr); // Doesn't work
    takes_layout(arr);
}

fn takes_arrref<A, D>(arr: &ArrayRef<A, D>) {}

fn takes_rawref<A, D>(arr: &RawRef<A, D>) {}

fn takes_layout<A, D>(arr: &LayoutRef<A, D>) {}

#[test]
fn tester()
{
    let arr = array![1, 2, 3];
    takes_base_raw(&arr);
    takes_arrref(&arr);
    takes_rawref(&arr); // Doesn't work
    takes_layout(&arr);
}
