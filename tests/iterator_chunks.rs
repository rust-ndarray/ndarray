
#[macro_use(s)]
extern crate ndarray;

use ndarray::prelude::*;
use ndarray::NdProducer;

#[test]
fn chunks() {
    let a = <Array1<f32>>::linspace(1., 100., 10 * 10).into_shape((10, 10)).unwrap();

    let (m, n) = a.dim();
    for i in 1..m + 1 {
        for j in 1..n + 1 {
            let c = a.whole_chunks((i, j));

            let ly = n / j;
            for (index, elt) in c.into_iter().enumerate() {
                assert_eq!(elt.dim(), (i, j));
                let cindex = (index / ly, index % ly);
                let cx = (cindex.0 * i) as isize;
                let cy = (cindex.1 * j) as isize;
                assert_eq!(elt, a.slice(s![cx.., cy..]).slice(s![..i as isize, ..j as isize]));
            }
            let c = a.whole_chunks((i, j));
            assert_eq!(c.into_iter().count(), (m / i) * (n / j));

            let c = a.whole_chunks((i, j));
            let (c1, c2) = c.split_at(Axis(0), (m / i) / 2);
            assert_eq!(c1.into_iter().count(), ((m / i) / 2) * (n / j));
            assert_eq!(c2.into_iter().count(), (m / i - (m / i) / 2) * (n / j));
        }
    }
    let c = a.whole_chunks((m + 1, n));
    assert_eq!(c.raw_dim().size(), 0);
    assert_eq!(c.into_iter().count(), 0);
}

#[should_panic]
#[test]
fn chunks_different_size_1() {
    let a = Array::<f32, _>::zeros(vec![2, 3]);
    a.whole_chunks(vec![2]);
}

#[test]
fn chunks_ok_size() {
    let mut a = Array::<f32, _>::zeros(vec![2, 3]);
    a.fill(1.);
    let mut c = 0;
    for elt in a.whole_chunks(vec![2, 1]) {
        assert!(elt.iter().all(|&x| x == 1.));
        assert_eq!(elt.dim(), vec![2, 1]);
        c += 1;
    }
    assert_eq!(c, 3);
}

#[should_panic]
#[test]
fn chunks_different_size_2() {
    let a = Array::<f32, _>::zeros(vec![2, 3]);
    a.whole_chunks(vec![2, 3, 4]);
}
