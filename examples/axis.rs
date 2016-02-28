extern crate ndarray;

use ndarray::{
    OwnedArray,
    Axis,
    Axis0, Axis1, Axis2,
};

fn main() {
    let mut a = OwnedArray::<f32, _>::linspace(0., 24., 25).into_shape((5, 5)).unwrap();
    println!("{:?}", a.subview(Axis0, 0));
    println!("{:?}", a.subview(Axis0, 1));
    println!("{:?}", a.subview(Axis1, 1));
    //println!("{:?}", a.subview(Axis2, 1));
    println!("{:?}", a.subview(Axis(0), 1));
}
