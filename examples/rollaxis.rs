use ndarray::prelude::*;
use ndarray::Data;

pub fn roll_axis<A, S, D>(mut a: ArrayBase<S, D>, to: Axis, from: Axis) -> ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    let i = to.index();
    let mut j = from.index();
    if j > i {
        while i != j {
            a.swap_axes(i, j);
            j -= 1;
        }
    } else {
        while i != j {
            a.swap_axes(i, j);
            j += 1;
        }
    }
    a
}

fn main() {
    let mut data = array![
        [[-1., 0., -2.], [1., 7., -3.]],
        [[1., 0., -3.], [1., 7., 5.]],
        [[1., 0., -3.], [1., 7., 5.]],
        [[2., 0., 2.], [1., 7., 2.]]
    ];

    println!("{:8.4?}", data);

    data = roll_axis(data, Axis(2), Axis(0));

    println!("{:8.4?}", data);
}
