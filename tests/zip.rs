use ndarray::{azip, ArrayBase, Data, DataMut, Dimension};

#[allow(unused)]
fn zip_ref_ref<Sa, Sb, D: Dimension>(mut a: &mut ArrayBase<Sa, D>, b: &ArrayBase<Sb, D>)
where Sa: DataMut<Elem = f64>,
      Sb: Data<Elem = f64>
{
    azip!(mut a, b in { *a = 2.0*b });
}
