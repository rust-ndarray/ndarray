use core::{
    array::IntoIter,
    iter::Cloned,
    marker::PhantomData,
    ops::{Add, AddAssign, Deref, DerefMut, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    slice::Iter,
};

use num_traits::Zero;

use crate::layout::{
    rank::{ConstRank, Rank},
    ranked::Ranked,
    shape::Shape,
    ShapeStrideError,
};

/// A wrapper for fixed-length arrays that can be used for shape and strides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeStrideN<T, const N: usize>([T; N]);

/// An array shape with a constant rank.
pub type NShape<const N: usize> = ShapeStrideN<usize, N>;

impl<T, const N: usize> Deref for ShapeStrideN<T, N>
{
    type Target = [T; N];

    fn deref(&self) -> &Self::Target
    {
        &self.0
    }
}

impl<T, const N: usize> DerefMut for ShapeStrideN<T, N>
{
    fn deref_mut(&mut self) -> &mut Self::Target
    {
        &mut self.0
    }
}

impl<T, const N: usize> Index<usize> for ShapeStrideN<T, N>
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output
    {
        &self.0[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for ShapeStrideN<T, N>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output
    {
        &mut self.0[index]
    }
}

impl<T, const N: usize> From<[T; N]> for ShapeStrideN<T, N>
{
    fn from(value: [T; N]) -> Self
    {
        Self { 0: value }
    }
}

impl<T, const N: usize> From<ShapeStrideN<T, N>> for [T; N]
{
    fn from(value: ShapeStrideN<T, N>) -> Self
    {
        value.0
    }
}

impl<'a, T, const N: usize> TryFrom<&'a [T]> for ShapeStrideN<T, N>
where T: Copy + Zero
{
    type Error = ShapeStrideError<Self>;

    fn try_from(value: &'a [T]) -> Result<Self, Self::Error>
    {
        if value.len() != N {
            Err(ShapeStrideError::RankMismatch(PhantomData, value.len()))
        } else {
            let mut arr = [T::zero(); N];
            arr.copy_from_slice(value);
            Ok(Self { 0: arr })
        }
    }
}

impl<T, const N: usize> IntoIterator for ShapeStrideN<T, N>
{
    type Item = T;

    type IntoIter = IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter
    {
        self.0.into_iter()
    }
}

macro_rules! shapestride_and_tuples {
    ($(($tuple:ty, $N:literal)),*) => {
        $(
            impl<T> From<$tuple> for ShapeStrideN<T, $N>
            {
                fn from(value: $tuple) -> Self
                {
                    Self { 0: value.into() }
                }
            }

            impl<T> From<ShapeStrideN<T, $N>> for $tuple
            {
                fn from(value: ShapeStrideN<T, $N>) -> Self
                {
                    value.0.into()
                }
            }
        )*
    };
}

shapestride_and_tuples!(
    ((T,), 1),
    ((T, T), 2),
    ((T, T, T), 3),
    ((T, T, T, T), 4),
    ((T, T, T, T, T), 5),
    ((T, T, T, T, T, T), 6),
    ((T, T, T, T, T, T, T), 7),
    ((T, T, T, T, T, T, T, T), 8),
    ((T, T, T, T, T, T, T, T, T), 9),
    ((T, T, T, T, T, T, T, T, T, T), 10),
    ((T, T, T, T, T, T, T, T, T, T, T), 11),
    ((T, T, T, T, T, T, T, T, T, T, T, T), 12)
);

impl<Rhs, T, const N: usize> PartialEq<Rhs> for ShapeStrideN<T, N>
where
    T: PartialEq,
    Rhs: AsRef<[T]>,
{
    fn eq(&self, other: &Rhs) -> bool
    {
        let other = other.as_ref();
        if other.len() != N {
            return false;
        }
        for i in 0..N {
            if self[i] != other[i] {
                return false;
            }
        }
        return true;
    }
}

impl<T, const N: usize> Ranked for ShapeStrideN<T, N>
where ConstRank<N>: Rank
{
    type NDim = ConstRank<N>;

    fn ndim(&self) -> usize
    {
        N
    }
}

impl<const N: usize> Zero for NShape<N>
{
    fn zero() -> Self
    {
        Self([usize::zero(); N])
    }

    fn is_zero(&self) -> bool
    {
        self.0.iter().all(|e| *e == 0usize)
    }
}

macro_rules! impl_op {
    ($op_trait:ty, $op_fn:ident, $op_assign_trait:ty, $op_assign_fn:ident) => {
        impl<Rhs, const N: usize> $op_trait for NShape<N>
        where
            Rhs: Into<NShape<N>>,
        {
            type Output = NShape<N>;

            fn $op_fn(self, rhs: Rhs) -> Self::Output {
                let mut output = self.clone();
                output.$op_assign_fn(rhs);
                output
            }
        }

        impl<Rhs, const N: usize> $op_assign_trait for NShape<N>
        where
            Rhs: Into<NShape<N>>,
        {
            fn $op_assign_fn(&mut self, rhs: Rhs) {
                let other = rhs.into();
                for i in 0..N {
                    self[i].$op_assign_fn(other[i]);
                }
            }
        }
    };
}

impl_op!(Add<Rhs>, add, AddAssign<Rhs>, add_assign);
impl_op!(Sub<Rhs>, sub, SubAssign<Rhs>, sub_assign);
impl_op!(Mul<Rhs>, mul, MulAssign<Rhs>, mul_assign);

impl<const N: usize> Add<usize> for NShape<N>
{
    type Output = NShape<N>;

    fn add(self, rhs: usize) -> Self::Output
    {
        let mut output = self.clone();
        output += rhs;
        output
    }
}

impl<const N: usize> AddAssign<usize> for NShape<N>
{
    fn add_assign(&mut self, rhs: usize)
    {
        for o in self.iter_mut() {
            *o += rhs;
        }
    }
}

impl<const N: usize> Mul<usize> for NShape<N>
{
    type Output = NShape<N>;

    fn mul(self, rhs: usize) -> Self::Output
    {
        let mut output = self.clone();
        for o in output.iter_mut() {
            *o = *o * rhs;
        }
        output
    }
}

impl<const N: usize> MulAssign<usize> for NShape<N>
{
    fn mul_assign(&mut self, rhs: usize)
    {
        for o in self.iter_mut() {
            *o += rhs;
        }
    }
}

impl<const N: usize> Shape for NShape<N>
where ConstRank<N>: Rank
{
    type Iter<'a> = Cloned<Iter<'a, usize>>;

    fn axis_len(&self, axis: usize) -> usize
    {
        self.0[axis]
    }

    fn iter(&self) -> Self::Iter<'_>
    {
        self.0.iter().cloned()
    }
}

impl<const N: usize> Default for NShape<N>
{
    fn default() -> Self
    {
        Self::zero()
    }
}
