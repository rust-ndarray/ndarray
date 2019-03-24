// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use std::fmt;
use super::{
    ArrayBase,
    Data,
    Dimension,
    NdProducer,
    Ix
};
use crate::dimension::IntoDimension;

#[derive(Debug)]
enum ArrayDisplayMode {
    // Array is small enough to me printed without omitting any values.
    Full,
    // Omit central values of the nth axis. Since we print that axis horizontally, ellipses
    // on each row do something like a split of the array into 2 parts vertically.
    VSplit,
    // Omit central values of the certain axis. Since we do it only once, ellipses on each row
    // do something like a split of the array into 2 parts horizontally.
    HSplit(Ix),
    // Both `VSplit` and `HSplit` hold.
    DoubleSplit(Ix),
}

const PRINT_ELEMENTS_LIMIT: Ix = 5;

impl ArrayDisplayMode {
    fn from_array<A, S, D>(arr: &ArrayBase<S, D>, limit: usize) -> ArrayDisplayMode
        where S: Data<Elem=A>,
              D: Dimension
    {
        let last_dim = arr.shape().len() - 1;

        let mut overflow_axis_pair: (Option<usize>, Option<usize>) = (None, None);
        for (axis, axis_size) in arr.shape().iter().enumerate().rev() {
            if *axis_size >= 2 * limit + 1 {
                match overflow_axis_pair.0 {
                    Some(_) => {
                        if let None = overflow_axis_pair.1 {
                            overflow_axis_pair.1 = Some(axis);
                        }
                    },
                    None => {
                        if axis != last_dim {
                            return ArrayDisplayMode::HSplit(axis);
                        }
                        overflow_axis_pair.0 = Some(axis);
                    }
                }
            }
        }

        match overflow_axis_pair {
            (Some(_), Some(h_axis)) => ArrayDisplayMode::DoubleSplit(h_axis),
            (Some(_), None) => ArrayDisplayMode::VSplit,
            (None, _) => ArrayDisplayMode::Full,
        }
    }

    fn h_split_offset(&self) -> Option<Ix> {
        match self {
            ArrayDisplayMode::DoubleSplit(axis) | ArrayDisplayMode::HSplit(axis) => {
                Some(axis + 1usize)
            },
            _ => None
        }
    }
}

fn format_array_v2<A, S, D, F>(view: &ArrayBase<S, D>,
                                  f: &mut fmt::Formatter,
                                  mut format: F,
                                  limit: usize) -> fmt::Result
    where F: FnMut(&A, &mut fmt::Formatter) -> fmt::Result,
          D: Dimension,
          S: Data<Elem=A>,
{
    let display_mode = ArrayDisplayMode::from_array(view, limit);

    let ndim = view.dim().into_dimension().slice().len();
    let nth_idx_max = view.shape().iter().last().unwrap();

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
    let mut printed_ellipses_h = false;
    // Shows if the row was printed for the first time after horizontal split.
    let mut no_rows_after_skip_yet = false;

    // Simply use the indexed iterator, and take the index wraparounds
    // as cues for when to add []'s and how many to add.
    for (index, elt) in view.indexed_iter() {
        let index = index.into_dimension();
        let take_n = if ndim == 0 { 1 } else { ndim - 1 };
        let mut update_index = false;

        let mut print_row = true;
        match display_mode {
            ArrayDisplayMode::HSplit(axis) | ArrayDisplayMode::DoubleSplit(axis) => {
                let sa_idx_max = view.shape().iter().skip(axis).next().unwrap();
                let sa_idx_val = index.slice().iter().skip(axis).next().unwrap();
                if sa_idx_val >= &limit && sa_idx_val < &(sa_idx_max - &limit) {
                    print_row = false;
                    no_rows_after_skip_yet = true;
                }
            },
            _ => {}
        }

        for (i, (a, b)) in index.slice()
            .iter()
            .take(take_n)
            .zip(last_index.slice().iter())
            .enumerate() {
            if a != b {
                if print_row {
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
                } else if !printed_ellipses_h {
                    let n = ndim - i - 1;
                    for _ in 0..n {
                        write!(f, "]")?;
                    }
                    write!(f, ",")?;
                    write!(f, "\n")?;
                    for _ in 0..display_mode.h_split_offset().unwrap() {
                        write!(f, " ")?;
                    }
                    write!(f, "...,\n")?;
                    printed_ellipses_h = true;
                }
                first = true;
                update_index = true;
                break;
            }
        }

        if print_row {
            let mut print_elt = true;
            let nth_idx_val = index.slice().iter().last().unwrap();
            match display_mode {
                ArrayDisplayMode::VSplit | ArrayDisplayMode::DoubleSplit(_) => {
                    if nth_idx_val >= &limit && nth_idx_val < &(nth_idx_max - &limit) {
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
