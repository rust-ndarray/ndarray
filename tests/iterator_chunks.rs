#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::float_cmp
)]

use ndarray::prelude::*;

#[test]
#[cfg(feature = "std")]
fn chunks() {
    use ndarray::NdProducer;
    let a = <Array1<f32>>::linspace(1., 100., 10 * 10)
        .into_shape((10, 10))
        .unwrap();

    let (m, n) = a.dim();
    for i in 1..=m {
        for j in 1..=n {
            let c = a.exact_chunks((i, j));

            let ly = n / j;
            for (index, elt) in c.into_iter().enumerate() {
                assert_eq!(elt.dim(), (i, j));
                let cindex = (index / ly, index % ly);
                let cx = (cindex.0 * i) as isize;
                let cy = (cindex.1 * j) as isize;
                assert_eq!(
                    elt,
                    a.slice(s![cx.., cy..])
                        .slice(s![..i as isize, ..j as isize])
                );
            }
            let c = a.exact_chunks((i, j));
            assert_eq!(c.into_iter().count(), (m / i) * (n / j));

            let c = a.exact_chunks((i, j));
            let (c1, c2) = c.split_at(Axis(0), (m / i) / 2);
            assert_eq!(c1.into_iter().count(), ((m / i) / 2) * (n / j));
            assert_eq!(c2.into_iter().count(), (m / i - (m / i) / 2) * (n / j));
        }
    }
    let c = a.exact_chunks((m + 1, n));
    assert_eq!(c.raw_dim().size(), 0);
    assert_eq!(c.into_iter().count(), 0);
}

#[should_panic]
#[test]
fn chunks_different_size_1() {
    let a = Array::<f32, _>::zeros(vec![2, 3]);
    a.exact_chunks(vec![2]);
}

#[test]
fn chunks_ok_size() {
    let mut a = Array::<f32, _>::zeros(vec![2, 3]);
    a.fill(1.);
    let mut c = 0;
    for elt in a.exact_chunks(vec![2, 1]) {
        assert!(elt.iter().all(|&x| x == 1.));
        assert_eq!(elt.shape(), &[2, 1]);
        c += 1;
    }
    assert_eq!(c, 3);
}

#[should_panic]
#[test]
fn chunks_different_size_2() {
    let a = Array::<f32, _>::zeros(vec![2, 3]);
    a.exact_chunks(vec![2, 3, 4]);
}

#[test]
fn chunks_mut() {
    let mut a = Array::zeros((7, 8));
    for (i, mut chunk) in a.exact_chunks_mut((2, 3)).into_iter().enumerate() {
        chunk.fill(i);
    }
    println!("{:?}", a);
    let ans = array![
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [2, 2, 2, 3, 3, 3, 0, 0],
        [2, 2, 2, 3, 3, 3, 0, 0],
        [4, 4, 4, 5, 5, 5, 0, 0],
        [4, 4, 4, 5, 5, 5, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ];
    assert_eq!(a, ans);
}

#[should_panic]
#[test]
fn chunks_different_size_3() {
    let mut a = Array::<f32, _>::zeros(vec![2, 3]);
    a.exact_chunks_mut(vec![2, 3, 4]);
}
