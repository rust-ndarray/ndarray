extern crate ndarray;

use ndarray::prelude::*;

fn std1d(a: ArrayView1<f64>) -> f64 {
    let n = a.len() as f64;
    if n == 0. {
        return 0.;
    }
    let mean = a.sum() / n;
    (a.fold(0., |acc, &x| acc + (x - mean).powi(2)) / n).sqrt()
}

fn std(a: &Array2<f64>, axis: Axis) -> Array1<f64> {
    a.map_axis(axis, std1d)
}

fn main() {
    // "recreating the following"
    // counts -= np.mean(counts, axis=0)
    // counts /= np.std(counts, axis=0)

    let mut data = array![[-1., -2., -3.], [1., -3., 5.], [2., 2., 2.]];

    println!("{:8.4}", data);
    println!("{:8.4} (Mean axis=0)", data.mean_axis(Axis(0)));

    data -= &data.mean_axis(Axis(0));
    println!("{:8.4}", data);

    data /= &std(&data, Axis(0));
    println!("{:8.4}", data);
}
