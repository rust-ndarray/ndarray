#[cfg(feature = "std")]
use ndarray::prelude::*;

#[cfg(feature = "std")]
fn main() {
    // This example recreates the following from python/numpy
    // counts -= np.mean(counts, axis=0)
    // counts /= np.std(counts, axis=0)

    let mut data = array![[-1., -2., -3.], [1., -3., 5.], [2., 2., 2.]];

    println!("{:8.4}", data);
    println!("Mean along axis=0 (along columns):\n{:8.4}", data.mean_axis(Axis(0)).unwrap());

    data -= &data.mean_axis(Axis(0)).unwrap();
    println!("Centered around mean:\n{:8.4}", data);

    data /= &data.std_axis(Axis(0), 0.);
    println!("Scaled to normalize std:\n{:8.4}", data);

    println!("New mean:\n{:8.4}", data.mean_axis(Axis(0)).unwrap());
    println!("New std: \n{:8.4}", data.std_axis(Axis(0), 0.));
}

#[cfg(not(feature = "std"))]
fn main() {}
