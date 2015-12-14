use std::fmt;
use super::{Array, Dimension};

fn format_array<A, D: Dimension, F>(view: &Array<A, D>, f: &mut fmt::Formatter,
                                    mut format: F) -> fmt::Result where
    F: FnMut(&mut fmt::Formatter, &A) -> fmt::Result,
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
        try!(write!(f, "["));
    }
    let mut first = true;
    // Simply use the indexed iterator, and take the index wraparounds
    // as cues for when to add []'s and how many to add.
    for (index, elt) in view.indexed_iter() {
        let take_n = if ndim == 0 { 1 } else { ndim - 1 };
        let mut update_index = false;
        for (i, (a, b)) in index.slice().iter().take(take_n)
                        .zip(last_index.slice().iter())
                        .enumerate()
        {
            if a != b {
                // New row.
                // # of ['s needed
                let n = ndim - i - 1;
                for _ in 0..n {
                    try!(write!(f, "]"));
                }
                try!(write!(f, ","));
                if !f.alternate() {
                    try!(write!(f, "\n"));
                }
                for _ in 0..ndim - n {
                    try!(write!(f, " "));
                }
                for _ in 0..n {
                    try!(write!(f, "["));
                }
                first = true;
                update_index = true;
                break;
            }
        }
        if !first {
            try!(write!(f, ", "));
        }
        first = false;
        try!(format(f, elt));

        if update_index {
            last_index = index;
        }
    }
    for _ in 0..ndim {
        try!(write!(f, "]"));
    }
    Ok(())
}

// NOTE: We can impl other fmt traits here
impl<'a, A: fmt::Display, D: Dimension> fmt::Display for Array<A, D>
{
    /// Format the array using `Display` and apply the formatting parameters used
    /// to each element.
    ///
    /// The array is shown in multiline style, unless the alternate form 
    /// is used -- i.e. `{:#}`.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        format_array(self, f, |f, elt| elt.fmt(f))
    }
}

impl<'a, A: fmt::Debug, D: Dimension> fmt::Debug for Array<A, D>
{
    /// Format the array using `Debug` and apply the formatting parameters used
    /// to each element.
    ///
    /// The array is shown in multiline style, unless the alternate form 
    /// is used -- i.e. `{:#}`.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        format_array(self, f, |f, elt| elt.fmt(f))
    }
}

impl<'a, A: fmt::LowerExp, D: Dimension> fmt::LowerExp for Array<A, D>
{
    /// Format the array using `LowerExp` and apply the formatting parameters used
    /// to each element.
    ///
    /// The array is shown in multiline style, unless the alternate form
    /// is used -- i.e. `{:#e}`.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        format_array(self, f, |f, elt| elt.fmt(f))
    }
}

impl<'a, A: fmt::UpperExp, D: Dimension> fmt::UpperExp for Array<A, D>
{
    /// Format the array using `UpperExp` and apply the formatting parameters used
    /// to each element.
    ///
    /// The array is shown in multiline style, unless the alternate form
    /// is used -- i.e. `{:#E}`.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        format_array(self, f, |f, elt| elt.fmt(f))
    }
}

