
use std::slice;

use libnum;
use imp_prelude::*;

/// ***Deprecated: Use `ArrayBase::zeros` instead.***
///
/// Return an array filled with zeros
#[cfg_attr(has_deprecated, deprecated(note="Use `ArrayBase::zeros` instead."))]
pub fn zeros<A, D>(dim: D) -> OwnedArray<A, D>
    where A: Clone + libnum::Zero, D: Dimension,
{
    ArrayBase::zeros(dim)
}

/// Return a zero-dimensional array with the element `x`.
pub fn arr0<A>(x: A) -> OwnedArray<A, ()>
{
    unsafe { ArrayBase::from_vec_dim_unchecked((), vec![x]) }
}

/// Return a one-dimensional array with elements from `xs`.
pub fn arr1<A: Clone>(xs: &[A]) -> OwnedArray<A, Ix> {
    ArrayBase::from_vec(xs.to_vec())
}

/// Return a one-dimensional array with elements from `xs`.
pub fn rcarr1<A: Clone>(xs: &[A]) -> RcArray<A, Ix> {
    arr1(xs).into_shared()
}

/// Return a zero-dimensional array view borrowing `x`.
pub fn aview0<A>(x: &A) -> ArrayView<A, ()> {
    unsafe { ArrayView::new_(x, (), ()) }
}

/// Return a one-dimensional array view with elements borrowing `xs`.
///
/// ```
/// use ndarray::aview1;
///
/// let data = [1.0; 1024];
///
/// // Create a 2D array view from borrowed data
/// let a2d = aview1(&data).into_shape((32, 32)).unwrap();
///
/// assert!(
///     a2d.scalar_sum() == 1024.0
/// );
/// ```
pub fn aview1<A>(xs: &[A]) -> ArrayView<A, Ix> {
    ArrayView::from_slice(xs)
}

/// Return a two-dimensional array view with elements borrowing `xs`.
pub fn aview2<A, V: FixedInitializer<Elem=A>>(xs: &[V]) -> ArrayView<A, (Ix, Ix)> {
    let cols = V::len();
    let rows = xs.len();
    let data = unsafe {
        slice::from_raw_parts(xs.as_ptr() as *const A, cols * rows)
    };
    let dim = (rows as Ix, cols as Ix);
    unsafe {
        let strides = dim.default_strides();
        ArrayView::new_(data.as_ptr(), dim, strides)
    }
}

/// Return a one-dimensional read-write array view with elements borrowing `xs`.
///
/// ```
/// #[macro_use(s)]
/// extern crate ndarray;
///
/// use ndarray::aview_mut1;
///
/// // Create an array view over some data, then slice it and modify it.
/// fn main() {
///     let mut data = [0; 1024];
///     {
///         let mut a = aview_mut1(&mut data).into_shape((32, 32)).unwrap();
///         a.slice_mut(s![.., ..;3]).assign_scalar(&5);
///     }
///     assert_eq!(&data[..10], [5, 0, 0, 5, 0, 0, 5, 0, 0, 5]);
/// }
/// ```
pub fn aview_mut1<A>(xs: &mut [A]) -> ArrayViewMut<A, Ix> {
    ArrayViewMut::from_slice(xs)
}

/// Fixed-size array used for array initialization
pub unsafe trait FixedInitializer {
    type Elem;
    fn as_init_slice(&self) -> &[Self::Elem];
    fn len() -> usize;
}

macro_rules! impl_arr_init {
    (__impl $n: expr) => (
        unsafe impl<T> FixedInitializer for [T;  $n] {
            type Elem = T;
            fn as_init_slice(&self) -> &[T] { self }
            fn len() -> usize { $n }
        }
    );
    () => ();
    ($n: expr, $($m:expr,)*) => (
        impl_arr_init!(__impl $n);
        impl_arr_init!($($m,)*);
    )

}

impl_arr_init!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,);

/// Return a two-dimensional array with elements from `xs`.
///
/// ```
/// use ndarray::arr2;
///
/// let a = arr2(&[[1, 2, 3],
///                [4, 5, 6]]);
/// assert!(
///     a.shape() == [2, 3]
/// );
/// ```
pub fn arr2<A: Clone, V: FixedInitializer<Elem = A>>(xs: &[V]) -> OwnedArray<A, (Ix, Ix)> {
    // FIXME: Simplify this when V is fix size array
    let (m, n) = (xs.len() as Ix,
                  xs.get(0).map_or(0, |snd| snd.as_init_slice().len() as Ix));
    let dim = (m, n);
    let mut result = Vec::<A>::with_capacity(dim.size());
    for snd in xs {
        let snd = snd.as_init_slice();
        result.extend(snd.iter().cloned());
    }
    unsafe {
        ArrayBase::from_vec_dim_unchecked(dim, result)
    }
}

/// Return a two-dimensional array with elements from `xs`.
///
pub fn rcarr2<A: Clone, V: FixedInitializer<Elem = A>>(xs: &[V]) -> RcArray<A, (Ix, Ix)> {
    arr2(xs).into_shared()
}

/// Return a three-dimensional array with elements from `xs`.
///
/// **Panics** if the slices are not all of the same length.
///
/// ```
/// use ndarray::arr3;
///
/// let a = arr3(&[[[1, 2],
///                 [3, 4]],
///                [[5, 6],
///                 [7, 8]],
///                [[9, 0],
///                 [1, 2]]]);
/// assert!(
///     a.shape() == [3, 2, 2]
/// );
/// ```
pub fn arr3<A: Clone, V: FixedInitializer<Elem=U>, U: FixedInitializer<Elem=A>>(xs: &[V])
    -> OwnedArray<A, (Ix, Ix, Ix)>
{
    // FIXME: Simplify this when U/V are fix size arrays
    let m = xs.len() as Ix;
    let fst = xs.get(0).map(|snd| snd.as_init_slice());
    let thr = fst.and_then(|elt| elt.get(0).map(|elt2| elt2.as_init_slice()));
    let n = fst.map_or(0, |v| v.len() as Ix);
    let o = thr.map_or(0, |v| v.len() as Ix);
    let dim = (m, n, o);
    let mut result = Vec::<A>::with_capacity(dim.size());
    for snd in xs {
        let snd = snd.as_init_slice();
        for thr in snd.iter() {
            let thr = thr.as_init_slice();
            result.extend(thr.iter().cloned());
        }
    }
    unsafe {
        ArrayBase::from_vec_dim_unchecked(dim, result)
    }
}

/// Return a three-dimensional array with elements from `xs`.
pub fn rcarr3<A: Clone, V: FixedInitializer<Elem=U>, U: FixedInitializer<Elem=A>>(xs: &[V])
    -> RcArray<A, (Ix, Ix, Ix)>
{
    arr3(xs).into_shared()
}
