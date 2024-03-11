use crate::error::*;
use crate::{Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

/// Calculate the common shape for a pair of array shapes, that they can be broadcasted
/// to. Return an error if the shapes are not compatible.
///
/// Uses the [NumPy broadcasting rules]
//  (https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules).
pub(crate) fn co_broadcast<D1, D2, Output>(shape1: &D1, shape2: &D2) -> Result<Output, ShapeError>
where
    D1: Dimension,
    D2: Dimension,
    Output: Dimension,
{
    let (k, overflow) = shape1.ndim().overflowing_sub(shape2.ndim());
    // Swap the order if d2 is longer.
    if overflow {
        return co_broadcast::<D2, D1, Output>(shape2, shape1);
    }
    // The output should be the same length as shape1.
    let mut out = Output::zeros(shape1.ndim());
    for (out, s) in izip!(out.slice_mut(), shape1.slice()) {
        *out = *s;
    }
    for (out, s2) in izip!(&mut out.slice_mut()[k..], shape2.slice()) {
        if *out != *s2 {
            if *out == 1 {
                *out = *s2
            } else if *s2 != 1 {
                return Err(from_kind(ErrorKind::IncompatibleShape));
            }
        }
    }
    Ok(out)
}

/// Return new stride when trying to grow `from` into shape `to`
///
/// Broadcasting works by returning a "fake stride" where elements
/// to repeat are in axes with 0 stride, so that several indexes point
/// to the same element.
///
/// **Note:** Cannot be used for mutable iterators, since repeating
/// elements would create aliasing pointers.
pub(crate) fn upcast<D: Dimension, E: Dimension>(to: &D, from: &E, stride: &E) -> Option<D> {
    // Make sure the product of non-zero axis lengths does not exceed
    // `isize::MAX`. This is the only safety check we need to perform
    // because all the other constraints of `ArrayBase` are guaranteed
    // to be met since we're starting from a valid `ArrayBase`.
    let _ = size_of_shape_checked(to).ok()?;

    let mut new_stride = to.clone();
    // begin at the back (the least significant dimension)
    // size of the axis has to either agree or `from` has to be 1
    if to.ndim() < from.ndim() {
        return None;
    }

    {
        let mut new_stride_iter = new_stride.slice_mut().iter_mut().rev();
        for ((er, es), dr) in from
            .slice()
            .iter()
            .rev()
            .zip(stride.slice().iter().rev())
            .zip(new_stride_iter.by_ref())
        {
            /* update strides */
            if *dr == *er {
                /* keep stride */
                *dr = *es;
            } else if *er == 1 {
                /* dead dimension, zero stride */
                *dr = 0
            } else {
                return None;
            }
        }

        /* set remaining strides to zero */
        for dr in new_stride_iter {
            *dr = 0;
        }
    }
    Some(new_stride)
}

pub trait DimMax<Other: Dimension> {
    /// The resulting dimension type after broadcasting.
    type Output: Dimension;
}

/// Dimensions of the same type remain unchanged when co_broadcast.
/// So you can directly use D as the resulting type.
/// (Instead of <D as DimMax<D>>::BroadcastOutput)
impl<D: Dimension> DimMax<D> for D {
    type Output = D;
}

macro_rules! impl_broadcast_distinct_fixed {
    ($smaller:ty, $larger:ty) => {
        impl DimMax<$larger> for $smaller {
            type Output = $larger;
        }

        impl DimMax<$smaller> for $larger {
            type Output = $larger;
        }
    };
}

impl_broadcast_distinct_fixed!(Ix0, Ix1);
impl_broadcast_distinct_fixed!(Ix0, Ix2);
impl_broadcast_distinct_fixed!(Ix0, Ix3);
impl_broadcast_distinct_fixed!(Ix0, Ix4);
impl_broadcast_distinct_fixed!(Ix0, Ix5);
impl_broadcast_distinct_fixed!(Ix0, Ix6);
impl_broadcast_distinct_fixed!(Ix1, Ix2);
impl_broadcast_distinct_fixed!(Ix1, Ix3);
impl_broadcast_distinct_fixed!(Ix1, Ix4);
impl_broadcast_distinct_fixed!(Ix1, Ix5);
impl_broadcast_distinct_fixed!(Ix1, Ix6);
impl_broadcast_distinct_fixed!(Ix2, Ix3);
impl_broadcast_distinct_fixed!(Ix2, Ix4);
impl_broadcast_distinct_fixed!(Ix2, Ix5);
impl_broadcast_distinct_fixed!(Ix2, Ix6);
impl_broadcast_distinct_fixed!(Ix3, Ix4);
impl_broadcast_distinct_fixed!(Ix3, Ix5);
impl_broadcast_distinct_fixed!(Ix3, Ix6);
impl_broadcast_distinct_fixed!(Ix4, Ix5);
impl_broadcast_distinct_fixed!(Ix4, Ix6);
impl_broadcast_distinct_fixed!(Ix5, Ix6);
impl_broadcast_distinct_fixed!(Ix0, IxDyn);
impl_broadcast_distinct_fixed!(Ix1, IxDyn);
impl_broadcast_distinct_fixed!(Ix2, IxDyn);
impl_broadcast_distinct_fixed!(Ix3, IxDyn);
impl_broadcast_distinct_fixed!(Ix4, IxDyn);
impl_broadcast_distinct_fixed!(Ix5, IxDyn);
impl_broadcast_distinct_fixed!(Ix6, IxDyn);


#[cfg(test)]
#[cfg(feature = "std")]
mod tests {
    use super::co_broadcast;
    use crate::{Dimension, Dim, DimMax, ShapeError, Ix0, IxDynImpl, ErrorKind};

    #[test]
    fn test_broadcast_shape() {
        fn test_co<D1, D2>(
            d1: &D1,
            d2: &D2,
            r: Result<<D1 as DimMax<D2>>::Output, ShapeError>,
        ) where
            D1: Dimension + DimMax<D2>,
            D2: Dimension,
        {
            let d = co_broadcast::<D1, D2, <D1 as DimMax<D2>>::Output>(&d1, d2);
            assert_eq!(d, r);
        }
        test_co(&Dim([2, 3]), &Dim([4, 1, 3]), Ok(Dim([4, 2, 3])));
        test_co(
            &Dim([1, 2, 2]),
            &Dim([1, 3, 4]),
            Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)),
        );
        test_co(&Dim([3, 4, 5]), &Ix0(), Ok(Dim([3, 4, 5])));
        let v = vec![1, 2, 3, 4, 5, 6, 7];
        test_co(
            &Dim(vec![1, 1, 3, 1, 5, 1, 7]),
            &Dim([2, 1, 4, 1, 6, 1]),
            Ok(Dim(IxDynImpl::from(v.as_slice()))),
        );
        let d = Dim([1, 2, 1, 3]);
        test_co(&d, &d, Ok(d));
        test_co(
            &Dim([2, 1, 2]).into_dyn(),
            &Dim(0),
            Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)),
        );
        test_co(
            &Dim([2, 1, 1]),
            &Dim([0, 0, 1, 3, 4]),
            Ok(Dim([0, 0, 2, 3, 4])),
        );
        test_co(&Dim([0]), &Dim([0, 0, 0]), Ok(Dim([0, 0, 0])));
        test_co(&Dim(1), &Dim([1, 0, 0]), Ok(Dim([1, 0, 0])));
        test_co(
            &Dim([1, 3, 0, 1, 1]),
            &Dim([1, 2, 3, 1]),
            Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)),
        );
    }
}
