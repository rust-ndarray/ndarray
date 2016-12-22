
#[macro_use(s)]
extern crate ndarray;

use ndarray::Array;
use ndarray::Dimension;
use ndarray::Axis;

fn regularize<A, D>(a: &mut Array<A, D>) -> Result<(), ()>
    where D: Dimension,
          A: ::std::fmt::Debug,
{
    println!("Regularize:\n{:?}", a);
    // reverse all neg axes
    while let Some(ax) = a.axes().find(|ax| ax.stride() <= 0) {
        if ax.stride() == 0 { return Err(()); }
        // reverse ax
        println!("Reverse {:?}", ax.axis());
        a.invert_axis(ax.axis());
    }

    // sort by least stride
    let mut i = 0;
    let n = a.ndim();
    while let Some(ax) = a.axes().rev().skip(i).min_by_key(|ax| ax.stride().abs()) {
        a.swap_axes(n - 1 - i, ax.axis().index());
        println!("Swap {:?} <=> {}", ax.axis(), n - 1 - i);
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
    println!("{:?}", a);
    Ok(())
}

fn main() {
    let mut a = Array::<u8, _>::zeros((2, 3, 4));
    for (i, elt) in (0..).zip(&mut a) {
        *elt = i;
    }
    a.swap_axes(0, 1);
    a.swap_axes(0, 2);
    a.islice(s![.., ..;-1, ..]);
    regularize(&mut a).ok();

    let mut b = Array::<u8, _>::zeros((2, 3, 4));
    for (i, elt) in (0..).zip(&mut b) {
        *elt = i;
    }
    regularize(&mut b).ok();
    let mut b = b.into_shape(a.len()).unwrap();
    regularize(&mut b).ok();
    b.invert_axis(Axis(0));
    regularize(&mut b).ok();

    let mut a = Array::<u8, _>::zeros((2, 3, 4));
    for (i, elt) in (0..).zip(&mut a) {
        *elt = i;
    }
    a.islice(s![..;-1, ..;2, ..]);
    regularize(&mut a).ok();
}
