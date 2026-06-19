use ndarray::prelude::*;
use ndarray::Zip;

#[test]
fn cell_view()
{
    let mut a = Array::from_shape_fn((10, 5), |(i, j)| (i * j) as f32);
    let answer = &a + 1.;

    {
        let cv1 = a.cell_view();
        let cv2 = cv1;

        Zip::from(cv1).and(cv2).for_each(|a, b| a.set(b.get() + 1.));
    }
    assert_eq!(a, answer);
}

#[test]
fn test_view_conversion()
{
    let mut a = Array2::<f32>::zeros((4, 4));
    let view_mut = a.view_mut();
    let view = view_mut.into_view();
    assert_eq!(view.shape(), &[4, 4]);

    let view_mut = a.view_mut();
    let view: ArrayView2<'_, f32> = view_mut.into();
    assert_eq!(view.shape(), &[4, 4]);
}

#[test]
fn test_view_conversion_lifetime()
{
    // Regression test for #1595
    struct Foo<'a>
    {
        data: ArrayViewMut2<'a, f32>,
    }

    impl<'a> Foo<'a>
    {
        fn into_shared(self) -> ArrayView2<'a, f32>
        {
            self.data.into_view()
        }

        fn into_shared_from(self) -> ArrayView2<'a, f32>
        {
            self.data.into()
        }
    }

    let mut a = Array2::<f32>::zeros((4, 4));
    let foo = Foo { data: a.view_mut() };
    let shared = foo.into_shared();
    assert_eq!(shared.shape(), &[4, 4]);

    let foo = Foo { data: a.view_mut() };
    let shared = foo.into_shared_from();
    assert_eq!(shared.shape(), &[4, 4]);
}
