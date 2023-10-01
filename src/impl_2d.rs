// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Methods for two-dimensional arrays.
use crate::imp_prelude::*;

/// # Methods For 2-D Arrays
impl<A, S> ArrayBase<S, Ix2>
where
    S: RawData<Elem = A>,
{
    /// Return an array view of row `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    ///
    /// ```
    /// use ndarray::array;
    /// let array = array![[1., 2.], [3., 4.]];
    /// assert_eq!(array.row(0), array![1., 2.]);
    /// ```
    pub fn row(&self, index: Ix) -> ArrayView1<'_, A>
    where
        S: Data,
    {
        self.index_axis(Axis(0), index)
    }

    /// Return a mutable array view of row `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    ///
    /// ```
    /// use ndarray::array;
    /// let mut array = array![[1., 2.], [3., 4.]];
    /// array.row_mut(0)[1] = 5.;
    /// assert_eq!(array, array![[1., 5.], [3., 4.]]);
    /// ```
    pub fn row_mut(&mut self, index: Ix) -> ArrayViewMut1<'_, A>
    where
        S: DataMut,
    {
        self.index_axis_mut(Axis(0), index)
    }

    /// Return the number of rows (length of `Axis(0)`) in the two-dimensional array.
    ///
    /// ```
    /// use ndarray::{array, Axis};
    ///
    /// let array = array![[1., 2.],
    ///                    [3., 4.],
    ///                    [5., 6.]];
    /// assert_eq!(array.nrows(), 3);
    ///
    /// // equivalent ways of getting the dimensions
    /// // get nrows, ncols by using dim:
    /// let (m, n) = array.dim();
    /// assert_eq!(m, array.nrows());
    /// // get length of any particular axis with .len_of()
    /// assert_eq!(m, array.len_of(Axis(0)));
    /// ```
    pub fn nrows(&self) -> usize {
        self.len_of(Axis(0))
    }

    /// Return an array view of column `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    ///
    /// ```
    /// use ndarray::array;
    /// let array = array![[1., 2.], [3., 4.]];
    /// assert_eq!(array.column(0), array![1., 3.]);
    /// ```
    pub fn column(&self, index: Ix) -> ArrayView1<'_, A>
    where
        S: Data,
    {
        self.index_axis(Axis(1), index)
    }

    /// Return a mutable array view of column `index`.
    ///
    /// **Panics** if `index` is out of bounds.
    ///
    /// ```
    /// use ndarray::array;
    /// let mut array = array![[1., 2.], [3., 4.]];
    /// array.column_mut(0)[1] = 5.;
    /// assert_eq!(array, array![[1., 2.], [5., 4.]]);
    /// ```
    pub fn column_mut(&mut self, index: Ix) -> ArrayViewMut1<'_, A>
    where
        S: DataMut,
    {
        self.index_axis_mut(Axis(1), index)
    }

    /// Return the number of columns (length of `Axis(1)`) in the two-dimensional array.
    ///
    /// ```
    /// use ndarray::{array, Axis};
    ///
    /// let array = array![[1., 2.],
    ///                    [3., 4.],
    ///                    [5., 6.]];
    /// assert_eq!(array.ncols(), 2);
    ///
    /// // equivalent ways of getting the dimensions
    /// // get nrows, ncols by using dim:
    /// let (m, n) = array.dim();
    /// assert_eq!(n, array.ncols());
    /// // get length of any particular axis with .len_of()
    /// assert_eq!(n, array.len_of(Axis(1)));
    /// ```
    pub fn ncols(&self) -> usize {
        self.len_of(Axis(1))
    }

    /// Return true if the array is square, false otherwise.
    ///
    /// # Examples
    /// Square:
    /// ```
    /// use ndarray::array;
    /// let array = array![[1., 2.], [3., 4.]];
    /// assert!(array.is_square());
    /// ```
    /// Not square:
    /// ```
    /// use ndarray::array;
    /// let array = array![[1., 2., 5.], [3., 4., 6.]];
    /// assert!(!array.is_square());
    /// ```
    pub fn is_square(&self) -> bool {
        let (m, n) = self.dim();
        m == n
    }

    /// Return the elements on and below the kth diagonal of A. 
    /// If k=-1, it returns the lower triangular part of A excluding the main diagonal. 
    /// If k=0 it returns the lower triangular part of A including the main diagonal. 
    /// If k=1 it returns the lower triangular part of A including the main diagonal and the diagonal above it, and so on
    /// 
    /// # Examples
    /// ```
    /// 
    /// use ndarray::{array, Array2};
    /// let A = Array2::<f64>::ones((3, 3));

    /// let L_1 = A.tril(-1);
    /// let L_2 = A.tril(0);
    /// let L_3 = A.tril(1);

    /// assert_eq!(
    ///     L_1,
    ///     array![
    ///         [0.0, 0.0, 0.0], 
    ///         [1.0, 0.0, 0.0], 
    ///         [1.0, 1.0, 0.0]
    ///     ]
    /// );
    /// assert_eq!(
    ///     L_2,
    ///     array![
    ///         [1.0, 0.0, 0.0], 
    ///         [1.0, 1.0, 0.0], 
    ///         [1.0, 1.0, 1.0]
    ///     ]
    /// );
    /// assert_eq!(
    ///     L_3,
    ///     array![
    ///         [1.0, 1.0, 0.0], 
    ///         [1.0, 1.0, 1.0], 
    ///         [1.0, 1.0, 1.0]
    ///     ]
    /// );
    /// 
    /// 
    /// ```
    pub fn tril (self, k:i32) -> Self
    where S: DataMut,
          A: num_traits::Zero,
    {
        let mut b: ArrayBase<S, Dim<[usize; 2]>> = self;
        let (n,m) = b.dim();
        for i in 0..n{
            for j in 0..m{
                match k.cmp(&0) {
                    std::cmp::Ordering::Less => {
                        if j +((- k) as usize)> i {
                            // A[[i,j]] = 0.0;
                            b.index_axis_mut(Axis(0), i)[j] = A::zero();
                        }
                    },
                    std::cmp::Ordering::Equal => {
                        if j > i{
                            b.index_axis_mut(Axis(0), i)[j] = A::zero();
                        }
                    },
                    std::cmp::Ordering::Greater => {
                        if j > i + k as usize{
                            b.index_axis_mut(Axis(0), i)[j] = A::zero();
                        }
                    }
                }
            }
        }
        b
    }

}
