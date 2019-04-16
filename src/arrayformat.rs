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

const PRINT_ELEMENTS_LIMIT: Ix = 3;

fn format_array<A, S, D, F>(view: &ArrayBase<S, D>,
                            f: &mut fmt::Formatter,
                            mut format: F,
                            limit: Ix) -> fmt::Result
    where F: FnMut(&A, &mut fmt::Formatter) -> fmt::Result,
          D: Dimension,
          S: Data<Elem=A>,
{
    if view.shape().is_empty() {
        // Handle 0-dimensional array case first
        return format(view.iter().next().unwrap(), f)
    }

    let overflow_axes: Vec<Ix> = view.shape().iter()
        .enumerate()
        .rev()
        .filter(|(_, axis_size)| **axis_size > 2 * limit)
        .map(|(axis, _)| axis)
        .collect();

    let ndim = view.ndim();
    let nth_idx_max = view.shape()[ndim-1];

    // None will be an empty iter.
    let mut last_index = match view.dim().into_dimension().first_index() {
        None => view.dim().into_dimension().clone(),
        Some(ix) => ix,
    };
    write!(f, "{}", "[".repeat(ndim))?;
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

        let skip_row_for_axis = overflow_axes.iter()
            .filter(|axis| {
                if **axis == ndim - 1 {
                    return false
                };
                let sa_idx_max = view.shape().iter().skip(**axis).next().unwrap();
                let sa_idx_val = index.slice().iter().skip(**axis).next().unwrap();
                sa_idx_val >= &limit && sa_idx_val < &(sa_idx_max - &limit)
            })
            .min()
            .map(|v| *v);
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
                        write!(f, "{}", "]".repeat(n))?;
                        writeln!(f, ",")?;
                    }
                    no_rows_after_skip_yet = false;
                    write!(f, "{}", " ".repeat(ndim - n))?;
                    write!(f, "{}", "[".repeat(n))?;
                } else if !printed_ellipses_h[skip_row_for_axis.unwrap()] {
                    let ax = skip_row_for_axis.unwrap();
                    let n = ndim - i - 1;
                    write!(f, "{}", "]".repeat(n))?;
                    writeln!(f, ",")?;
                    write!(f, "{}", " ".repeat(ax + 1))?;
                    writeln!(f, "...,")?;
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
            if overflow_axes.contains(&(ndim - 1)) {
                let nth_idx_val = nth_idx_op.unwrap();
                if nth_idx_val >= &limit && nth_idx_val < &(nth_idx_max - &limit) {
                    print_elt = false;
                    if !printed_ellipses_v {
                        write!(f, ", ...")?;
                        printed_ellipses_v = true;
                    }
                }
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
    write!(f, "{}", "]".repeat(ndim))?;
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
        format_array(self, f, <_>::fmt, PRINT_ELEMENTS_LIMIT)
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
        format_array(self, f, <_>::fmt, PRINT_ELEMENTS_LIMIT)?;
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
        format_array(self, f, <_>::fmt, PRINT_ELEMENTS_LIMIT)
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
        format_array(self, f, <_>::fmt, PRINT_ELEMENTS_LIMIT)
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
        format_array(self, f, <_>::fmt, PRINT_ELEMENTS_LIMIT)
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
        format_array(self, f, <_>::fmt, PRINT_ELEMENTS_LIMIT)
    }
}

#[cfg(test)]
mod formatting_with_omit {
    use crate::prelude::*;
    use super::*;

    fn print_output_diff(expected: &str, actual: &str) {
        println!("Expected output:\n{}\nActual output:\n{}", expected, actual);
    }

    #[test]
    fn dim_1() {
        let overflow: usize = 5;
        let a = Array1::from_elem((PRINT_ELEMENTS_LIMIT * 2 + overflow, ), 1);
        let mut expected_output = String::from("[");
        a.iter()
            .take(PRINT_ELEMENTS_LIMIT)
            .for_each(|elem| { expected_output.push_str(format!("{}, ", elem).as_str()) });
        expected_output.push_str("...");
        a.iter()
            .skip(PRINT_ELEMENTS_LIMIT + overflow)
            .for_each(|elem| { expected_output.push_str(format!(", {}", elem).as_str()) });
        expected_output.push(']');
        let actual_output = format!("{}", a);

        print_output_diff(&expected_output, &actual_output);
        assert_eq!(actual_output, expected_output);
    }

    #[test]
    fn dim_2_last_axis_overflow() {
        let overflow: usize = 3;
        let a = Array2::from_elem((PRINT_ELEMENTS_LIMIT, PRINT_ELEMENTS_LIMIT * 2 + overflow), 1);
        let mut expected_output = String::from("[");

        for i in 0..PRINT_ELEMENTS_LIMIT {
            expected_output.push_str(format!("[{}", a[(i, 0)]).as_str());
            for j in 1..PRINT_ELEMENTS_LIMIT {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str(", ...");
            for j in PRINT_ELEMENTS_LIMIT + overflow..PRINT_ELEMENTS_LIMIT * 2 + overflow {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str(if i < PRINT_ELEMENTS_LIMIT - 1 { "],\n " } else { "]" });
        }
        expected_output.push(']');
        let actual_output = format!("{}", a);

        print_output_diff(&expected_output, &actual_output);
        assert_eq!(actual_output, expected_output);
    }

    #[test]
    fn dim_2_non_last_axis_overflow() {
        let overflow: usize = 5;
        let a = Array2::from_elem((PRINT_ELEMENTS_LIMIT * 2 + overflow, PRINT_ELEMENTS_LIMIT), 1);
        let mut expected_output = String::from("[");

        for i in 0..PRINT_ELEMENTS_LIMIT {
            expected_output.push_str(format!("[{}", a[(i, 0)]).as_str());
            for j in 1..PRINT_ELEMENTS_LIMIT {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str("],\n ");
        }
        expected_output.push_str("...,\n ");
        for i in PRINT_ELEMENTS_LIMIT + overflow..PRINT_ELEMENTS_LIMIT * 2 + overflow {
            expected_output.push_str(format!("[{}", a[(i, 0)]).as_str());
            for j in 1..PRINT_ELEMENTS_LIMIT {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str(if i == PRINT_ELEMENTS_LIMIT * 2 + overflow - 1 {
                "]"
            } else {
                "],\n "
            });
        }
        expected_output.push(']');
        let actual_output = format!("{}", a);

        print_output_diff(&expected_output, &actual_output);
        assert_eq!(actual_output, expected_output);
    }

    #[test]
    fn dim_2_multi_directional_overflow() {
        let overflow: usize = 5;
        let a = Array2::from_elem(
            (PRINT_ELEMENTS_LIMIT * 2 + overflow, PRINT_ELEMENTS_LIMIT * 2 + overflow), 1
        );
        let mut expected_output = String::from("[");

        for i in 0..PRINT_ELEMENTS_LIMIT {
            expected_output.push_str(format!("[{}", a[(i, 0)]).as_str());
            for j in 1..PRINT_ELEMENTS_LIMIT {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str(", ...");
            for j in PRINT_ELEMENTS_LIMIT + overflow..PRINT_ELEMENTS_LIMIT * 2 + overflow {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str("],\n ");
        }
        expected_output.push_str("...,\n ");
        for i in PRINT_ELEMENTS_LIMIT + overflow..PRINT_ELEMENTS_LIMIT * 2 + overflow {
            expected_output.push_str(format!("[{}", a[(i, 0)]).as_str());
            for j in 1..PRINT_ELEMENTS_LIMIT {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str(", ...");
            for j in PRINT_ELEMENTS_LIMIT + overflow..PRINT_ELEMENTS_LIMIT * 2 + overflow {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str(if i == PRINT_ELEMENTS_LIMIT * 2 + overflow - 1 {
                "]"
            } else {
                "],\n "
            });
        }
        expected_output.push(']');
        let actual_output = format!("{}", a);

        print_output_diff(&expected_output, &actual_output);
        assert_eq!(actual_output, expected_output);
    }
}
