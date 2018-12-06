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
};
use crate::dimension::IntoDimension;

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
        format_array(self, f, <_>::fmt)
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
        format_array(self, f, <_>::fmt)?;
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
