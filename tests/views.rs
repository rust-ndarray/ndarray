use ndarray::prelude::*;
use ndarray::Zip;

#[test]
fn cell_view() {
    let mut a = Array::from_shape_fn((10, 5), |(i, j)| (i * j) as f32);
    let answer = &a + 1.;

    {
        let cv1 = a.cell_view();
        let cv2 = cv1;

        Zip::from(cv1).and(cv2).for_each(|a, b| a.set(b.get() + 1.));
    }
    assert_eq!(a, answer);
}
