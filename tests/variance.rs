use ndarray::{Array1, ArrayView1};

fn arrayview_covariant<'a: 'b, 'b>(x: ArrayView1<'a, f64>) -> ArrayView1<'b, f64>
{
    x
}

#[test]
fn test_covariance()
{
    let x = Array1::zeros(2);
    let shorter_view = arrayview_covariant(x.view());
    assert_eq!(shorter_view[0], 0.0);
}
