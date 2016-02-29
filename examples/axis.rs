extern crate ndarray;

use ndarray::{
    OwnedArray,
    Axis,
};

fn main() {
    let a = OwnedArray::<f32, _>::linspace(0., 24., 25).into_shape((5, 5)).unwrap();
    println!("{:?}", a.subview(Axis(0), 0));
    println!("{:?}", a.subview(Axis(0), 1));
    println!("{:?}", a.subview(Axis(1), 1));
    println!("{:?}", a.subview(Axis(0), 1));
    println!("{:?}", a.subview(Axis(2), 1)); // PANIC
}
