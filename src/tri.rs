// Copyright 2014-2024 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::cmp::{max, min};

use num_traits::Zero;

use crate::{dimension::is_layout_f, Array, ArrayBase, Axis, Data, Dimension, IntoDimension, Zip};

impl<S, A, D> ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
    A: Clone + Zero,
    D::Smaller: Copy,
{
    /// Upper triangular of an array.
    ///
    /// Return a copy of the array with elements below the *k*-th diagonal zeroed.
    /// For arrays with `ndim` exceeding 2, `triu` will apply to the final two axes.
    /// For 0D and 1D arrays, `triu` will return an unchanged clone.
    ///
    /// See also [`ArrayBase::tril`]
    ///
    /// ```
    /// use ndarray::array;
    ///
    /// let arr = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// let res = arr.triu(0);
    /// assert_eq!(res, array![[1, 2, 3], [0, 5, 6], [0, 0, 9]]);
    /// ```
    pub fn triu(&self, k: isize) -> Array<A, D>
    {
        if self.ndim() <= 1 {
            return self.to_owned();
        }
        match is_layout_f(&self.dim, &self.strides) {
            true => {
                let n = self.ndim();
                let mut x = self.view();
                x.swap_axes(n - 2, n - 1);
                let mut tril = x.tril(-k);
                tril.swap_axes(n - 2, n - 1);

                tril
            }
            false => {
                let mut res = Array::zeros(self.raw_dim());
                Zip::indexed(self.rows())
                    .and(res.rows_mut())
                    .for_each(|i, src, mut dst| {
                        let row_num = i.into_dimension().last_elem();
                        let lower = max(row_num as isize + k, 0);
                        dst.slice_mut(s![lower..]).assign(&src.slice(s![lower..]));
                    });

                res
            }
        }
    }

    /// Lower triangular of an array.
    ///
    /// Return a copy of the array with elements above the *k*-th diagonal zeroed.
    /// For arrays with `ndim` exceeding 2, `tril` will apply to the final two axes.
    /// For 0D and 1D arrays, `tril` will return an unchanged clone.
    ///
    /// See also [`ArrayBase::triu`]
    ///
    /// ```
    /// use ndarray::array;
    ///
    /// let arr = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// let res = arr.tril(0);
    /// assert_eq!(res, array![[1, 0, 0], [4, 5, 0], [7, 8, 9]]);
    /// ```
    pub fn tril(&self, k: isize) -> Array<A, D>
    {
        if self.ndim() <= 1 {
            return self.to_owned();
        }
        match is_layout_f(&self.dim, &self.strides) {
            true => {
                let n = self.ndim();
                let mut x = self.view();
                x.swap_axes(n - 2, n - 1);
                let mut tril = x.triu(-k);
                tril.swap_axes(n - 2, n - 1);

                tril
            }
            false => {
                let mut res = Array::zeros(self.raw_dim());
                let ncols = self.len_of(Axis(self.ndim() - 1)) as isize;
                Zip::indexed(self.rows())
                    .and(res.rows_mut())
                    .for_each(|i, src, mut dst| {
                        let row_num = i.into_dimension().last_elem();
                        let upper = min(row_num as isize + k, ncols) + 1;
                        dst.slice_mut(s![..upper]).assign(&src.slice(s![..upper]));
                    });

                res
            }
        }
    }
}

#[cfg(test)]
mod tests
{
    use crate::{array, dimension, Array0, Array1, Array2, Array3, ShapeBuilder};
    use alloc::vec;

    #[test]
    fn test_keep_order()
    {
        let x = Array2::<f64>::ones((3, 3).f());
        let res = x.triu(0);
        assert!(dimension::is_layout_f(&res.dim, &res.strides));

        let res = x.tril(0);
        assert!(dimension::is_layout_f(&res.dim, &res.strides));
    }

    #[test]
    fn test_0d()
    {
        let x = Array0::<f64>::ones(());
        let res = x.triu(0);
        assert_eq!(res, x);

        let res = x.tril(0);
        assert_eq!(res, x);

        let x = Array0::<f64>::ones(().f());
        let res = x.triu(0);
        assert_eq!(res, x);

        let res = x.tril(0);
        assert_eq!(res, x);
    }

    #[test]
    fn test_1d()
    {
        let x = array![1, 2, 3];
        let res = x.triu(0);
        assert_eq!(res, x);

        let res = x.triu(0);
        assert_eq!(res, x);

        let x = Array1::<f64>::ones(3.f());
        let res = x.triu(0);
        assert_eq!(res, x);

        let res = x.triu(0);
        assert_eq!(res, x);
    }

    #[test]
    fn test_2d()
    {
        let x = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];

        // Upper
        let res = x.triu(0);
        assert_eq!(res, array![[1, 2, 3], [0, 5, 6], [0, 0, 9]]);

        // Lower
        let res = x.tril(0);
        assert_eq!(res, array![[1, 0, 0], [4, 5, 0], [7, 8, 9]]);

        let x = Array2::from_shape_vec((3, 3).f(), vec![1, 4, 7, 2, 5, 8, 3, 6, 9]).unwrap();

        // Upper
        let res = x.triu(0);
        assert_eq!(res, array![[1, 2, 3], [0, 5, 6], [0, 0, 9]]);

        // Lower
        let res = x.tril(0);
        assert_eq!(res, array![[1, 0, 0], [4, 5, 0], [7, 8, 9]]);
    }

    #[test]
    fn test_3d()
    {
        let x = array![
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
        ];

        // Upper
        let res = x.triu(0);
        assert_eq!(
            res,
            array![
                [[1, 2, 3], [0, 5, 6], [0, 0, 9]],
                [[10, 11, 12], [0, 14, 15], [0, 0, 18]],
                [[19, 20, 21], [0, 23, 24], [0, 0, 27]]
            ]
        );

        // Lower
        let res = x.tril(0);
        assert_eq!(
            res,
            array![
                [[1, 0, 0], [4, 5, 0], [7, 8, 9]],
                [[10, 0, 0], [13, 14, 0], [16, 17, 18]],
                [[19, 0, 0], [22, 23, 0], [25, 26, 27]]
            ]
        );

        let x = Array3::from_shape_vec(
            (3, 3, 3).f(),
            vec![1, 10, 19, 4, 13, 22, 7, 16, 25, 2, 11, 20, 5, 14, 23, 8, 17, 26, 3, 12, 21, 6, 15, 24, 9, 18, 27],
        )
        .unwrap();

        // Upper
        let res = x.triu(0);
        assert_eq!(
            res,
            array![
                [[1, 2, 3], [0, 5, 6], [0, 0, 9]],
                [[10, 11, 12], [0, 14, 15], [0, 0, 18]],
                [[19, 20, 21], [0, 23, 24], [0, 0, 27]]
            ]
        );

        // Lower
        let res = x.tril(0);
        assert_eq!(
            res,
            array![
                [[1, 0, 0], [4, 5, 0], [7, 8, 9]],
                [[10, 0, 0], [13, 14, 0], [16, 17, 18]],
                [[19, 0, 0], [22, 23, 0], [25, 26, 27]]
            ]
        );
    }

    #[test]
    fn test_off_axis()
    {
        let x = array![
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
        ];

        let res = x.triu(1);
        assert_eq!(
            res,
            array![
                [[0, 2, 3], [0, 0, 6], [0, 0, 0]],
                [[0, 11, 12], [0, 0, 15], [0, 0, 0]],
                [[0, 20, 21], [0, 0, 24], [0, 0, 0]]
            ]
        );

        let res = x.triu(-1);
        assert_eq!(
            res,
            array![
                [[1, 2, 3], [4, 5, 6], [0, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [0, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [0, 26, 27]]
            ]
        );
    }

    #[test]
    fn test_odd_shape()
    {
        let x = array![[1, 2, 3], [4, 5, 6]];
        let res = x.triu(0);
        assert_eq!(res, array![[1, 2, 3], [0, 5, 6]]);

        let x = array![[1, 2], [3, 4], [5, 6]];
        let res = x.triu(0);
        assert_eq!(res, array![[1, 2], [0, 4], [0, 0]]);
    }
}
