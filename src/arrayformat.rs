// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use std::fmt;
use std::slice::Iter;
use super::{
    ArrayBase,
    Data,
    Dimension,
    NdProducer,
    Ix
};
use crate::dimension::IntoDimension;

#[derive(Debug, PartialEq)]
enum ArrayDisplayMode {
    // Array is small enough to be printed without omitting any values.
    Full,
    // Omit central values of the nth axis.
    OmitV,
    // Omit central values of certain axes (but not the last one).
    // Vector is guaranteed to be non-empty.
    OmitH(Vec<Ix>),
    // Both `OmitV` and `OmitH` hold.
    OmitBoth(Vec<Ix>),
}

const PRINT_ELEMENTS_LIMIT: Ix = 2;

impl ArrayDisplayMode {

    fn from_shape(shape: &[Ix], limit: Ix) -> ArrayDisplayMode {
        let last_dim = match shape.len().checked_sub(1) {
            Some(v) => v,
            None => {
                return ArrayDisplayMode::Full;
            }
        };

        let last_axis_ovf = shape[last_dim] >= 2 * limit + 1;
        let mut overflow_axes: Vec<Ix> = Vec::with_capacity(shape.len());
        for (axis, axis_size) in shape.iter().enumerate().rev() {
            if axis == last_dim {
                continue;
            }
            if *axis_size >= 2 * limit + 1 {
                overflow_axes.push(axis);
            }
        }

        if !overflow_axes.is_empty() && last_axis_ovf {
            ArrayDisplayMode::OmitBoth(overflow_axes)
        } else if !overflow_axes.is_empty() {
            ArrayDisplayMode::OmitH(overflow_axes)
        } else if last_axis_ovf {
            ArrayDisplayMode::OmitV
        } else {
            ArrayDisplayMode::Full
        }
    }

    fn h_axes_iter(&self) -> Option<Iter<Ix>> {
        match self {
            ArrayDisplayMode::OmitH(v) | ArrayDisplayMode::OmitBoth(v) => {
                Some(v.iter())
            },
            _ => None
        }
    }
}

fn format_array_v2<A, S, D, F>(view: &ArrayBase<S, D>,
                               f: &mut fmt::Formatter,
                               mut format: F,
                               limit: Ix) -> fmt::Result
    where F: FnMut(&A, &mut fmt::Formatter) -> fmt::Result,
          D: Dimension,
          S: Data<Elem=A>,
{
    let display_mode = ArrayDisplayMode::from_shape(view.shape(), limit);

    let ndim = view.dim().into_dimension().slice().len();
    let nth_idx_max = if ndim > 0 { Some(view.shape().iter().last().unwrap()) } else { None };

    // None will be an empty iter.
    let mut last_index = match view.dim().into_dimension().first_index() {
        None => view.dim().into_dimension().clone(),
        Some(ix) => ix,
    };
    for _ in 0..ndim {
        write!(f, "[")?;
    }
    let mut first = true;
    // Shows if ellipses for vertical split were printed.
    let mut printed_ellipses_v = false;
    // Shows if ellipses for horizontal split were printed.
    let mut printed_ellipses_h = vec![false; ndim];
    // Shows if the row was printed for the first time after horizontal split.
    let mut no_rows_after_skip_yet = false;

    // Simply use the indexed iterator, and take the index wraparounds
    // as cues for when to add []'s and how many to add.
    for (index, elt) in view.indexed_iter() {
        let index = index.into_dimension();
        let take_n = if ndim == 0 { 1 } else { ndim - 1 };
        let mut update_index = false;

        let skip_row_for_axis = match display_mode.h_axes_iter() {
            Some(iter) => {
                iter.filter(|axis| {
                    let sa_idx_max = view.shape().iter().skip(**axis).next().unwrap();
                    let sa_idx_val = index.slice().iter().skip(**axis).next().unwrap();
                    sa_idx_val >= &limit && sa_idx_val < &(sa_idx_max - &limit)
                })
                    .min()
                    .map(|v| *v)
            },
            None => None
        };
        if let Some(_) = skip_row_for_axis {
            no_rows_after_skip_yet = true;
        }

        for (i, (a, b)) in index.slice()
            .iter()
            .take(take_n)
            .zip(last_index.slice().iter())
            .enumerate() {
            if a != b {
                printed_ellipses_h.iter_mut().skip(i + 1).for_each(|e| { *e = false; });

                if skip_row_for_axis.is_none() {
                    printed_ellipses_v = false;
                    // New row.
                    // # of ['s needed
                    let n = ndim - i - 1;
                    if !no_rows_after_skip_yet {
                        for _ in 0..n {
                            write!(f, "]")?;
                        }
                        write!(f, ",")?;
                        write!(f, "\n")?;
                    }
                    no_rows_after_skip_yet = false;
                    for _ in 0..ndim - n {
                        write!(f, " ")?;
                    }
                    for _ in 0..n {
                        write!(f, "[")?;
                    }
                } else if !printed_ellipses_h[skip_row_for_axis.unwrap()] {
                    let ax = skip_row_for_axis.unwrap();
                    let n = ndim - i - 1;
                    for _ in 0..n {
                        write!(f, "]")?;
                    }
                    write!(f, ",")?;
                    write!(f, "\n")?;
                    for _ in 0..(ax + 1) {
                        write!(f, " ")?;
                    }
                    write!(f, "...,\n")?;
                    printed_ellipses_h[ax] = true;
                }
                first = true;
                update_index = true;
                break;
            }
        }

        if skip_row_for_axis.is_none() {
            let mut print_elt = true;
            let nth_idx_op = index.slice().iter().last();
            match display_mode {
                ArrayDisplayMode::OmitV | ArrayDisplayMode::OmitBoth(_) => {
                    let nth_idx_val = nth_idx_op.unwrap();
                    if nth_idx_val >= &limit && nth_idx_val < &(nth_idx_max.unwrap() - &limit) {
                        print_elt = false;
                        if !printed_ellipses_v {
                            write!(f, ", ...")?;
                            printed_ellipses_v = true;
                        }
                    }
                }
                _ => {}
            }

            if print_elt {
                if !first {
                    write!(f, ", ")?;
                }
                first = false;
                format(elt, f)?;
            }
        }

        if update_index {
            last_index = index;
        }
    }
    for _ in 0..ndim {
        write!(f, "]")?;
    }
    Ok(())
}

fn format_array<A, S, D, F>(view: &ArrayBase<S, D>, f: &mut fmt::Formatter,
                            mut format: F)
    -> fmt::Result
    where F: FnMut(&A, &mut fmt::Formatter) -> fmt::Result,
          D: Dimension,
          S: Data<Elem=A>,
{
    let ndim = view.dim.slice().len();
    /* private nowadays
    if ndim > 0 && f.width.is_none() {
        f.width = Some(4)
    }
    */
    // None will be an empty iter.
    let mut last_index = match view.dim.first_index() {
        None => view.dim.clone(),
        Some(ix) => ix,
    };
    for _ in 0..ndim {
        write!(f, "[")?;
    }
    let mut first = true;
    // Simply use the indexed iterator, and take the index wraparounds
    // as cues for when to add []'s and how many to add.
    for (index, elt) in view.indexed_iter() {
        let index = index.into_dimension();
        let take_n = if ndim == 0 { 1 } else { ndim - 1 };
        let mut update_index = false;
        for (i, (a, b)) in index.slice()
                                .iter()
                                .take(take_n)
                                .zip(last_index.slice().iter())
                                .enumerate() {
            if a != b {
                // New row.
                // # of ['s needed
                let n = ndim - i - 1;
                for _ in 0..n {
                    write!(f, "]")?;
                }
                write!(f, ",")?;
                write!(f, "\n")?;
                for _ in 0..ndim - n {
                    write!(f, " ")?;
                }
                for _ in 0..n {
                    write!(f, "[")?;
                }
                first = true;
                update_index = true;
                break;
            }
        }
        if !first {
            write!(f, ", ")?;
        }
        first = false;
        format(elt, f)?;

        if update_index {
            last_index = index;
        }
    }
    for _ in 0..ndim {
        write!(f, "]")?;
    }
    Ok(())
}

// NOTE: We can impl other fmt traits here
/// Format the array using `Display` and apply the formatting parameters used
/// to each element.
///
/// The array is shown in multiline style.
impl<'a, A: fmt::Display, S, D: Dimension> fmt::Display for ArrayBase<S, D>
    where S: Data<Elem=A>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        format_array_v2(self, f, <_>::fmt, PRINT_ELEMENTS_LIMIT)
    }
}

/// Format the array using `Debug` and apply the formatting parameters used
/// to each element.
///
/// The array is shown in multiline style.
impl<'a, A: fmt::Debug, S, D: Dimension> fmt::Debug for ArrayBase<S, D>
    where S: Data<Elem=A>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Add extra information for Debug
        format_array_v2(self, f, <_>::fmt, PRINT_ELEMENTS_LIMIT)?;
        write!(f, " shape={:?}, strides={:?}, layout={:?}",
               self.shape(), self.strides(), layout=self.view().layout())?;
        match D::NDIM {
            Some(ndim) => write!(f, ", const ndim={}", ndim)?,
            None => write!(f, ", dynamic ndim={}", self.ndim())?,
        }
        Ok(())
    }
}

/// Format the array using `LowerExp` and apply the formatting parameters used
/// to each element.
///
/// The array is shown in multiline style.
impl<'a, A: fmt::LowerExp, S, D: Dimension> fmt::LowerExp for ArrayBase<S, D>
    where S: Data<Elem=A>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        format_array(self, f, <_>::fmt)
    }
}

/// Format the array using `UpperExp` and apply the formatting parameters used
/// to each element.
///
/// The array is shown in multiline style.
impl<'a, A: fmt::UpperExp, S, D: Dimension> fmt::UpperExp for ArrayBase<S, D>
    where S: Data<Elem=A>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        format_array(self, f, <_>::fmt)
    }
}
/// Format the array using `LowerHex` and apply the formatting parameters used
/// to each element.
///
/// The array is shown in multiline style.
impl<'a, A: fmt::LowerHex, S, D: Dimension> fmt::LowerHex for ArrayBase<S, D>
    where S: Data<Elem=A>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        format_array(self, f, <_>::fmt)
    }
}

/// Format the array using `Binary` and apply the formatting parameters used
/// to each element.
///
/// The array is shown in multiline style.
impl<'a, A: fmt::Binary, S, D: Dimension> fmt::Binary for ArrayBase<S, D>
    where S: Data<Elem=A>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        format_array(self, f, <_>::fmt)
    }
}

#[cfg(test)]
mod format_tests {
    use super::*;

    #[test]
    fn test_array_display_mode_from_shape() {
        let mode = ArrayDisplayMode::from_shape(&[4, 4], 2);
        assert_eq!(mode, ArrayDisplayMode::Full);

        let mode = ArrayDisplayMode::from_shape(&[3, 6], 2);
        assert_eq!(mode, ArrayDisplayMode::OmitV);

        let mode = ArrayDisplayMode::from_shape(&[5, 6, 3], 2);
        assert_eq!(mode, ArrayDisplayMode::OmitH(vec![1, 0]));
    }
}
