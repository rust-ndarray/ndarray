
extern crate ndarray;

use ndarray::{
    OwnedArray,
    ArrayBase,
    DataMut,
    Dimension,
    ArrayView,
};

use std::cmp;

#[derive(Copy, Clone, Debug)]
pub enum ZipError {
    NotSameLayout,
    NotSameShape,
}

/// Defines a function similar to zip_mut_with, that takes multiple read-only
/// arguments. All arrays must be of default layout and same shape.
macro_rules! define_zip {
    ($name:ident, $($arg:ident),+) => {
#[allow(non_snake_case)]
fn $name<A, $($arg,)+ Data, Dim, Func>(a: &mut ArrayBase<Data, Dim>,
    $($arg: ArrayView<$arg, Dim>,)+ mut f: Func)
    -> Result<(), ZipError>
    where Data: DataMut<Elem=A>,
          Dim: Dimension,
          Func: FnMut(&mut A, $(&$arg),+)
{
    if $(a.shape() != $arg.shape() ||)+ false {
        return Err(ZipError::NotSameShape);
    }
    if let Some(a_s) = a.as_slice_mut() {
        let len = a_s.len();
        $(
            // extract the slice
            let $arg = if let Some(s) = $arg.as_slice() {
                s
            } else {
                return Err(ZipError::NotSameLayout);
            };
            let len = cmp::min(len, $arg.len());
        )+
        let a_s = &mut a_s[..len];
        $(
            let $arg = &$arg[..len];
        )+
        for i in 0..len {
            f(&mut a_s[i], $(&$arg[i]),+)
        }
        return Ok(());
    }
    // otherwise
    Err(ZipError::NotSameLayout)
}
    }
}

define_zip!(zip_2, B);
define_zip!(zip_3, B, C);
define_zip!(zip_4, B, C, D);
define_zip!(zip_5, B, C, D, E);


fn main() {
    let n = 16;
    let mut a = OwnedArray::<f32, _>::zeros((n, n));
    let mut b = OwnedArray::<f32, _>::from_elem((n, n), 1.);
    for ((i, j), elt) in b.indexed_iter_mut() {
        *elt /= 1. + (i + j) as f32;
    }
    let c = OwnedArray::<f32, _>::from_elem((n, n), 1.7);

    for _ in 0..1000 {
        zip_3(&mut a, b.view(), c.view(), |x, &y, &z| *x += y * z).unwrap();
    }
    println!("{:4.2?}", a);
}
