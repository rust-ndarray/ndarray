
use ndarray::prelude::*;

use ndarray::stack;

use num_complex::Complex;

fn cx<T>(re: T, im: T) -> Complex<T> {
    Complex::new(re, im)
}

#[test]
fn test_expand_from_zero() {
    let a = Array::from_elem((), [[[1, 2], [3, 4], [5, 6]],
                                  [[11, 12], [13, 14], [15, 16]]]);
    let av = a.view();
    println!("{:?}", av);
    let av = av.expand(Axis(0));
    println!("{:?}", av);
    let av = av.expand(Axis(1));
    println!("{:?}", av);
    let av = av.expand(Axis(2));
    println!("{:?}", av);
    assert!(av.is_standard_layout());
    assert_eq!(av, av.to_owned());

    let av = a.view();
    println!("{:?}", av);
    let av = av.expand(Axis(0));
    println!("{:?}", av);
    let av = av.expand(Axis(0));
    println!("{:?}", av);
    let av = av.expand(Axis(0));
    println!("{:?}", av);
    assert!(av.t().is_standard_layout());
    assert_eq!(av, av.to_owned());
}

#[test]
fn test_expand_zero() {
    let a = Array::from_elem((3, 4), [0.; 0]);

    for ax in 0..=2 {
        let mut new_shape = [3, 4, 4];
        new_shape[1] = if ax == 0 { 3 } else { 4 };
        new_shape[ax] = 0;
        let av = a.view();
        let av = av.expand(Axis(ax));
        assert_eq!(av.shape(), &new_shape);
    }
}

#[test]
fn test_expand1() {
    let a = Array::from_elem((3, 3), [1, 2, 3]);
    println!("{:?}", a);
    let b = a.view().expand(Axis(2));
    println!("{:?}", b);
    let b = a.view().expand(Axis(1));
    println!("{:?}", b);
    let b = a.view().expand(Axis(0));
    println!("{:?}", b);
}


#[test]
fn test_complex() {
    let a = arr2(&[[cx(3., 4.), cx(2., 0.)], [cx(0., -2.), cx(3., 0.)]]);
    let av = a.view();
    for ax in 0..=2 {
        let av = av.expand(Axis(ax));
        let answer = stack![Axis(ax), a.mapv(|z| z.re), a.mapv(|z| z.im)];
        assert_eq!(av, answer);
    }
}
