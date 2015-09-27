
extern crate ndarray;

use ndarray::Array;
//use ndarray::{Si, S};

fn main()
{
    let mat = Array::range(0.0f32, 16.0).reshape_clone((2, 4, 2));
    println!("{a:?}\n times \n{b:?}\nis equal to:\n{c:?}",
             a=mat.subview(2,1),
             b=mat.subview(0,1),
             c=mat.subview(2,1).mat_mul(&mat.subview(0,1)));

}
