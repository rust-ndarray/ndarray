use ndarray::{arr1, ArcArray1, Array1, ArrayView1, ArrayViewMut1, CowArray, Ix1, RawArrayView};

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

// The aliases must stay covariant when used as struct fields (#1610). Struct
// variance is computed from field types without normalizing projections, so an
// alias that leaves `A` to its `<S as RawData>::Elem` default would make the
// containing struct invariant (rust-lang/rust#115799).
//
// Each `shorten_*` function compiles only if the struct is covariant.
// `RawArrayViewMut` is omitted: like `*mut T`, it is correctly invariant.

struct HoldsView<'a>(ArrayView1<'a, f64>);

fn shorten_view<'a: 'b, 'b>(x: HoldsView<'a>) -> HoldsView<'b>
{
    x
}

struct HoldsViewMut<'a>(ArrayViewMut1<'a, f64>);

fn shorten_view_mut<'a: 'b, 'b>(x: HoldsViewMut<'a>) -> HoldsViewMut<'b>
{
    x
}

struct HoldsCow<'a>(CowArray<'a, f64, Ix1>);

fn shorten_cow<'a: 'b, 'b>(x: HoldsCow<'a>) -> HoldsCow<'b>
{
    x
}

struct HoldsViewOfRefs<'a>(ArrayView1<'a, &'a str>);

fn shorten_view_of_refs<'a: 'b, 'b>(x: HoldsViewOfRefs<'a>) -> HoldsViewOfRefs<'b>
{
    x
}

struct HoldsArray<T>(Array1<T>);

fn shorten_array<'a: 'b, 'b>(x: HoldsArray<&'a str>) -> HoldsArray<&'b str>
{
    x
}

struct HoldsArcArray<T>(ArcArray1<T>);

fn shorten_arc_array<'a: 'b, 'b>(x: HoldsArcArray<&'a str>) -> HoldsArcArray<&'b str>
{
    x
}

struct HoldsRawView<T>(RawArrayView<T, Ix1>);

fn shorten_raw_view<'a: 'b, 'b>(x: HoldsRawView<&'a str>) -> HoldsRawView<&'b str>
{
    x
}

#[test]
fn test_covariance_in_struct()
{
    let mut a = arr1(&[1., 2.]);
    assert_eq!(shorten_view(HoldsView(a.view())).0[0], 1.);
    assert_eq!(shorten_cow(HoldsCow(CowArray::from(a.view()))).0[1], 2.);
    {
        let HoldsViewMut(mut v) = shorten_view_mut(HoldsViewMut(a.view_mut()));
        v[0] = 3.;
    }
    assert_eq!(a[0], 3.);

    let s = String::from("hello");
    let strs = arr1(&[s.as_str()]);
    assert_eq!(shorten_view_of_refs(HoldsViewOfRefs(strs.view())).0[0], "hello");
    assert_eq!(shorten_array(HoldsArray(strs.clone())).0[0], "hello");
    assert_eq!(shorten_arc_array(HoldsArcArray(strs.clone().into_shared())).0[0], "hello");
    assert_eq!(shorten_raw_view(HoldsRawView(strs.raw_view())).0.len(), 1);
}
