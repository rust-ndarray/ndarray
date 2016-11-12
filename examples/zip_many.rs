
extern crate ndarray;

use ndarray::{
    Array,
    ArrayBase,
    Data,
    DataMut,
    Dimension,
    ArrayView,
    ArrayViewMut,
    Ix,
};

use std::cmp;

#[derive(Copy, Clone, Debug)]
pub enum ZipError {
    NotSameLayout,
    NotSameShape,
}

pub trait Slice {
    type Ref ;
    fn len(&self) -> usize;
    unsafe fn get_unchecked(&mut self, i: usize) -> Self::Ref ;
}

impl<'a, T> Slice for &'a [T]
{
    type Ref  = &'a T;
    fn len(&self) -> usize { (**self).len() }
    unsafe fn get_unchecked(&mut self, i: usize) -> Self::Ref  {
        (*self).get_unchecked(i)
    }
}

impl<'a, T> Slice for &'a mut [T]
{
    type Ref  = &'a mut T;
    fn len(&self) -> usize { (**self).len() }
    unsafe fn get_unchecked(&mut self, i: usize) -> Self::Ref  {
        (*(*self as *mut [T])).get_unchecked_mut(i)
    }
}

pub trait LockStep {
    type Item;
    type Ref;
    type Slice: Slice<Ref=Self::Ref>;
    type Dim;
    fn shape(&self) -> &[Ix];
    fn borrow_slice(self) -> Option<Self::Slice>;
}

impl<'b, A: 'b, S,  D> LockStep for &'b ArrayBase<S, D>
    where D: Dimension,
          S: Data<Elem=A>,
{
    type Item = A;
    type Ref = &'b A;
    type Slice = &'b [A];
    type Dim = D;
    fn shape(&self) -> &[Ix] { (*self).shape() }
    fn borrow_slice(self) -> Option<Self::Slice> {
        self.as_slice()
    }
}

impl<'b, A: 'b, S,  D> LockStep for &'b mut ArrayBase<S, D>
    where D: Dimension,
          S: DataMut<Elem=A>,
{
    type Item = A;
    type Ref = &'b mut A;
    type Slice = &'b mut [A];
    type Dim = D;
    fn shape(&self) -> &[Ix] { (**self).shape() }
    fn borrow_slice(self) -> Option<Self::Slice> {
        self.as_slice_mut()
    }
}

impl<'a, A, D> LockStep for ArrayView<'a, A, D>
    where D: Dimension,
{
    type Item = A;
    type Ref = &'a A;
    type Slice = &'a [A];
    type Dim = D;
    fn shape(&self) -> &[Ix] { self.shape() }
    fn borrow_slice(self) -> Option<Self::Slice> {
        self.into_slice()
    }
}

impl<'a, A, D> LockStep for ArrayViewMut<'a, A, D>
    where D: Dimension,
{
    type Item = A;
    type Ref = &'a mut A;
    type Slice = &'a mut [A];
    type Dim = D;
    fn shape(&self) -> &[Ix] { self.shape() }
    fn borrow_slice(self) -> Option<Self::Slice> {
        self.into_slice()
    }
}

/// Defines a function similar to zip_mut_with, that takes multiple read-only
/// arguments. All arrays must be of default layout and same shape.
macro_rules! define_zip {
    ($name:ident, $($arg:ident),+) => {
#[allow(non_snake_case)]
fn $name<A, $($arg),+, Dim, Func>(a: A, $($arg : $arg,)+ mut f: Func)
    -> Result<(), ZipError>
    where Dim: Dimension,
          A: LockStep<Dim=Dim>,
          $(
              $arg : LockStep<Dim=Dim>,
          )+
          Func: FnMut(A::Ref, $($arg ::Ref),+)
{
    if $(a.shape() != $arg.shape() ||)+ false {
        return Err(ZipError::NotSameShape);
    }
    if let Some(mut a_s) = a.borrow_slice() {
        let len = a_s.len();
        $(
            // extract the slice
            let mut $arg = if let Some(s) = $arg.borrow_slice() {
                s
            } else {
                return Err(ZipError::NotSameLayout);
            };
            let len = cmp::min(len, $arg.len());
        )+
        for i in 0..len {
            unsafe {
                f(a_s.get_unchecked(i), $($arg.get_unchecked(i)),+)
            }
        }
        return Ok(());
    }
    // otherwise
    Err(ZipError::NotSameLayout)
}
    }
}

//define_zip!(zip_2, B);
define_zip!(zip_3, B, C);
//define_zip!(zip_4, B, C, D);
//define_zip!(zip_5, B, C, D, E);


fn main() {
    let n = 16;
    let mut a = Array::<f32, _>::zeros((n, n));
    let mut b = Array::<f32, _>::from_elem((n, n), 1.);
    for ((i, j), elt) in b.indexed_iter_mut() {
        *elt /= 1. + (i + j) as f32;
    }
    let c = Array::<f32, _>::from_elem((n, n), 1.7);

    for _ in 0..1000 {
        zip_3(&mut a, &b, &c, |x, &y, &z| *x += y * z).unwrap();
    }
    println!("{:4.2?}", a);
}
