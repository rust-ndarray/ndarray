extern crate ndarray;

use ndarray::{
    OwnedArray,
    Ax,
    Ax0, Ax1, Ax2,
    AxisForDimension,
    Ix,
};
use ndarray::{Ax as Axis, Ax0 as Axis0, Ax1 as Axis1};

fn main() {
    let mut a = OwnedArray::<f32, _>::linspace(0., 24., 25).into_shape((5, 5)).unwrap();
    let x: AxisForDimension<(Ix, Ix)> = Ax(2).into();
    let x: AxisForDimension<(Ix, Ix)> = Ax0.into();
    let x: AxisForDimension<(Ix, Ix)> = Ax1.into();
    //let x: AxisForDimension<(Ix, Ix)> = Ax2.into();
    println!("{:?}", x);
    println!("{:?}", a.subview(Axis0, 0));
    println!("{:?}", a.subview(Axis0, 1));
    println!("{:?}", a.subview(Axis1, 1));
    println!("{:?}", a.subview(Axis(0), 1));
}
