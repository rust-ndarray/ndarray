
use libnum;
use itertools::free::enumerate;

use imp_prelude::*;
use numeric_util;

use {
    LinalgScalar,
};

impl<A, S> ArrayBase<S, Ix>
    where S: Data<Elem=A>,
{
    /// Compute the dot product of one-dimensional arrays.
    ///
    /// The dot product is a sum of the elementwise products (no conjugation
    /// of complex operands, and thus not their inner product).
    ///
    /// **Panics** if the arrays are not of the same length.
    pub fn dot<S2>(&self, rhs: &ArrayBase<S2, Ix>) -> A
        where S2: Data<Elem=A>,
              A: LinalgScalar,
    {
        self.dot_impl(rhs)
    }

    fn dot_generic<S2>(&self, rhs: &ArrayBase<S2, Ix>) -> A
        where S2: Data<Elem=A>,
              A: LinalgScalar,
    {
        debug_assert_eq!(self.len(), rhs.len());
        assert!(self.len() == rhs.len());
        if let Some(self_s) = self.as_slice() {
            if let Some(rhs_s) = rhs.as_slice() {
                return numeric_util::unrolled_dot(self_s, rhs_s);
            }
        }
        let mut sum = A::zero();
        for i in 0..self.len() {
            unsafe {
                sum = sum.clone() + self.uget(i).clone() * rhs.uget(i).clone();
            }
        }
        sum
    }

    #[cfg(not(feature="rblas"))]
    fn dot_impl<S2>(&self, rhs: &ArrayBase<S2, Ix>) -> A
        where S2: Data<Elem=A>,
              A: LinalgScalar,
    {
        self.dot_generic(rhs)
    }

    #[cfg(feature="rblas")]
    fn dot_impl<S2>(&self, rhs: &ArrayBase<S2, Ix>) -> A
        where S2: Data<Elem=A>,
              A: LinalgScalar,
    {
        use std::any::{Any, TypeId};
        use rblas::vector::ops::Dot;
        use linalg::AsBlasAny;

        // Read pointer to type `A` as type `B`.
        //
        // **Panics** if `A` and `B` are not the same type
        fn cast_as<A: Any + Copy, B: Any + Copy>(a: &A) -> B {
            assert_eq!(TypeId::of::<A>(), TypeId::of::<B>());
            unsafe {
                ::std::ptr::read(a as *const _ as *const B)
            }
        }
        // Use only if the vector is large enough to be worth it
        if self.len() >= 32 {
            debug_assert_eq!(self.len(), rhs.len());
            assert!(self.len() == rhs.len());
            if let Ok(self_v) = self.blas_view_as_type::<f32>() {
                if let Ok(rhs_v) = rhs.blas_view_as_type::<f32>() {
                    let f_ret = f32::dot(&self_v, &rhs_v);
                    return cast_as::<f32, A>(&f_ret);
                }
            }
            if let Ok(self_v) = self.blas_view_as_type::<f64>() {
                if let Ok(rhs_v) = rhs.blas_view_as_type::<f64>() {
                    let f_ret = f64::dot(&self_v, &rhs_v);
                    return cast_as::<f64, A>(&f_ret);
                }
            }
        }
        self.dot_generic(rhs)
    }
}


impl<A, S> ArrayBase<S, (Ix, Ix)>
    where S: Data<Elem=A>,
{
    /// Return an array view of row `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn row(&self, index: Ix) -> ArrayView<A, Ix>
    {
        self.subview(Axis(0), index)
    }

    /// Return a mutable array view of row `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn row_mut(&mut self, index: Ix) -> ArrayViewMut<A, Ix>
        where S: DataMut
    {
        self.subview_mut(Axis(0), index)
    }

    /// Return an array view of column `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn column(&self, index: Ix) -> ArrayView<A, Ix>
    {
        self.subview(Axis(1), index)
    }

    /// Return a mutable array view of column `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    pub fn column_mut(&mut self, index: Ix) -> ArrayViewMut<A, Ix>
        where S: DataMut
    {
        self.subview_mut(Axis(1), index)
    }

    /// Perform matrix multiplication of rectangular arrays `self` and `rhs`.
    ///
    /// The array shapes must agree in the way that
    /// if `self` is *M* × *N*, then `rhs` is *N* × *K*.
    ///
    /// Return a result array with shape *M* × *K*.
    ///
    /// **Panics** if shapes are incompatible.
    ///
    /// ```
    /// use ndarray::arr2;
    ///
    /// let a = arr2(&[[1., 2.],
    ///                [0., 1.]]);
    /// let b = arr2(&[[1., 2.],
    ///                [2., 3.]]);
    ///
    /// assert!(
    ///     a.mat_mul(&b) == arr2(&[[5., 8.],
    ///                             [2., 3.]])
    /// );
    /// ```
    ///
    pub fn mat_mul(&self, rhs: &ArrayBase<S, (Ix, Ix)>) -> OwnedArray<A, (Ix, Ix)>
        where A: LinalgScalar,
    {
        // NOTE: Matrix multiplication only defined for Copy types to
        // avoid trouble with panicking + and *, and destructors

        let ((m, a), (b, n)) = (self.dim, rhs.dim);
        let (self_columns, other_rows) = (a, b);
        assert!(self_columns == other_rows);

        // Avoid initializing the memory in vec -- set it during iteration
        // Panic safe because A: Copy
        let mut res_elems = Vec::<A>::with_capacity(m as usize * n as usize);
        unsafe {
            res_elems.set_len(m as usize * n as usize);
        }
        let mut i = 0;
        let mut j = 0;
        for rr in &mut res_elems {
            unsafe {
                *rr = (0..a).fold(libnum::zero::<A>(),
                    move |s, k| s + *self.uget((i, k)) * *rhs.uget((k, j))
                );
            }
            j += 1;
            if j == n {
                j = 0;
                i += 1;
            }
        }
        unsafe {
            ArrayBase::from_vec_dim_unchecked((m, n), res_elems)
        }
    }

    /// Perform the matrix multiplication of the rectangular array `self` and
    /// column vector `rhs`.
    ///
    /// The array shapes must agree in the way that
    /// if `self` is *M* × *N*, then `rhs` is *N*.
    ///
    /// Return a result array with shape *M*.
    ///
    /// **Panics** if shapes are incompatible.
    pub fn mat_mul_col(&self, rhs: &ArrayBase<S, Ix>) -> OwnedArray<A, Ix>
        where A: LinalgScalar,
    {
        let ((m, a), n) = (self.dim, rhs.dim);
        let (self_columns, other_rows) = (a, n);
        assert!(self_columns == other_rows);

        // Avoid initializing the memory in vec -- set it during iteration
        let mut res_elems = Vec::<A>::with_capacity(m as usize);
        unsafe {
            res_elems.set_len(m as usize);
        }
        for (i, rr) in enumerate(&mut res_elems) {
            unsafe {
                *rr = (0..a).fold(libnum::zero::<A>(),
                    move |s, k| s + *self.uget((i, k)) * *rhs.uget(k)
                );
            }
        }
        unsafe {
            ArrayBase::from_vec_dim_unchecked(m, res_elems)
        }
    }
}


