#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names
)]

use ndarray::prelude::*;

/// Reorder a's axes so that they are in "standard" axis order;
/// make sure axes are in positive stride direction, and merge adjacent
/// axes if possible.
///
/// This changes the logical order of the elements in the
/// array, so that if we read them in row-major order after regularization,
/// it corresponds to their order in memory.
///
/// Errors if array has a 0-stride axis
fn regularize<A, D>(a: &mut Array<A, D>) -> Result<(), &'static str>
where
    D: Dimension,
    A: ::std::fmt::Debug,
{
    println!("Regularize:\n{:?}", a);
    // reverse all neg axes
    while let Some(ax) = a.axes().find(|ax| ax.stride <= 0) {
        if ax.stride == 0 {
            // no real reason to error on this case; other than
            // stride == 0 is incompatible with an owned array.
            return Err("Cannot regularize array with stride == 0 axis");
        }
        // reverse ax
        println!("Reverse {:?}", ax.axis);
        a.invert_axis(ax.axis);
    }

    // sort by least stride
    let mut i = 0;
    let n = a.ndim();
    while let Some(ax) = a.axes().rev().skip(i).min_by_key(|ax| ax.stride.abs()) {
        let cur_axis = Axis(n - 1 - i);
        if ax.axis != cur_axis {
            a.swap_axes(cur_axis.index(), ax.axis.index());
            println!("Swap {:?} <=> {:?}", cur_axis, ax.axis);
        }
        i += 1;
    }

    // merge the lower axes if possible
    for j in (0..n).rev().skip(1) {
        if a.merge_axes(Axis(j), Axis(n - 1)) {
            println!("Merged {:?} into {:?}", Axis(j), Axis(n - 1));
        } else {
            break;
        }
    }
    println!("Result:\n{:?}\n", a);
    Ok(())
}

fn main() {
    let mut a = Array::<u8, _>::zeros((2, 3, 4));
    for (i, elt) in (0..).zip(&mut a) {
        *elt = i;
    }
    a.swap_axes(0, 1);
    a.swap_axes(0, 2);
    a.slice_collapse(s![.., ..;-1, ..]);
    regularize(&mut a).unwrap();

    let mut b = Array::<u8, _>::zeros((2, 3, 4));
    for (i, elt) in (0..).zip(&mut b) {
        *elt = i;
    }
    regularize(&mut b).unwrap();

    let mut b = b.into_shape(a.len()).unwrap();
    regularize(&mut b).unwrap();

    b.invert_axis(Axis(0));
    regularize(&mut b).unwrap();

    let mut a = Array::<u8, _>::zeros((2, 3, 4));
    for (i, elt) in (0..).zip(&mut a) {
        *elt = i;
    }
    a.slice_collapse(s![..;-1, ..;2, ..]);
    regularize(&mut a).unwrap();
}
