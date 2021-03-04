#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::float_cmp
)]

use ndarray::prelude::*;
use ndarray::Zip;

use itertools::{assert_equal, cloned};

use std::mem::swap;

#[test]
fn test_azip1() {
    let mut a = Array::zeros(62);
    let mut x = 0;
    azip!((a in &mut a) { *a = x; x += 1; });
    assert_equal(cloned(&a), 0..a.len());
}

#[test]
fn test_azip2() {
    let mut a = Array::zeros((5, 7));
    let b = Array::from_shape_fn(a.dim(), |(i, j)| 1. / (i + 2 * j) as f32);
    azip!((a in &mut a, &b in &b) *a = b);
    assert_eq!(a, b);
}

#[test]
fn test_azip2_1() {
    let mut a = Array::zeros((5, 7));
    let b = Array::from_shape_fn((5, 10), |(i, j)| 1. / (i + 2 * j) as f32);
    let b = b.slice(s![..;-1, 3..]);
    azip!((a in &mut a, &b in &b) *a = b);
    assert_eq!(a, b);
}

#[test]
fn test_azip2_3() {
    let mut b = Array::from_shape_fn((5, 10), |(i, j)| 1. / (i + 2 * j) as f32);
    let mut c = Array::from_shape_fn((5, 10), |(i, j)| f32::exp((i + j) as f32));
    let a = b.clone();
    azip!((b in &mut b, c in &mut c) swap(b, c));
    assert_eq!(a, c);
    assert!(a != b);
}

#[test]
#[cfg(feature = "approx")]
fn test_zip_collect() {
    use approx::assert_abs_diff_eq;

    // test Zip::map_collect and that it preserves c/f layout.

    let b = Array::from_shape_fn((5, 10), |(i, j)| 1. / (i + 2 * j + 1) as f32);
    let c = Array::from_shape_fn((5, 10), |(i, j)| f32::exp((i + j) as f32));

    {
        let a = Zip::from(&b).and(&c).map_collect(|x, y| x + y);

        assert_abs_diff_eq!(a, &b + &c, epsilon = 1e-6);
        assert_eq!(a.strides(), b.strides());
    }

    {
        let b = b.t();
        let c = c.t();

        let a = Zip::from(&b).and(&c).map_collect(|x, y| x + y);

        assert_abs_diff_eq!(a, &b + &c, epsilon = 1e-6);
        assert_eq!(a.strides(), b.strides());
    }
}

#[test]
#[cfg(feature = "approx")]
fn test_zip_assign_into() {
    use approx::assert_abs_diff_eq;

    let mut a = Array::<f32, _>::zeros((5, 10));
    let b = Array::from_shape_fn((5, 10), |(i, j)| 1. / (i + 2 * j + 1) as f32);
    let c = Array::from_shape_fn((5, 10), |(i, j)| f32::exp((i + j) as f32));

    Zip::from(&b).and(&c).map_assign_into(&mut a, |x, y| x + y);

    assert_abs_diff_eq!(a, &b + &c, epsilon = 1e-6);
}

#[test]
#[cfg(feature = "approx")]
fn test_zip_assign_into_cell() {
    use approx::assert_abs_diff_eq;
    use std::cell::Cell;

    let a = Array::<Cell<f32>, _>::default((5, 10));
    let b = Array::from_shape_fn((5, 10), |(i, j)| 1. / (i + 2 * j + 1) as f32);
    let c = Array::from_shape_fn((5, 10), |(i, j)| f32::exp((i + j) as f32));

    Zip::from(&b).and(&c).map_assign_into(&a, |x, y| x + y);
    let a2 = a.mapv(|elt| elt.get());

    assert_abs_diff_eq!(a2, &b + &c, epsilon = 1e-6);
}

#[test]
fn test_zip_collect_drop() {
    use std::cell::RefCell;
    use std::panic;

    struct Recorddrop<'a>((usize, usize), &'a RefCell<Vec<(usize, usize)>>);

    impl<'a> Drop for Recorddrop<'a> {
        fn drop(&mut self) {
            self.1.borrow_mut().push(self.0);
        }
    }

    #[derive(Copy, Clone)]
    enum Config {
        CC,
        CF,
        FF,
    }

    impl Config {
        fn a_is_f(self) -> bool {
            match self {
                Config::CC | Config::CF => false,
                _ => true,
            }
        }
        fn b_is_f(self) -> bool {
            match self {
                Config::CC => false,
                _ => true,
            }
        }
    }

    let test_collect_panic = |config: Config, will_panic: bool, slice: bool| {
        let mut inserts = RefCell::new(Vec::new());
        let mut drops = RefCell::new(Vec::new());

        let mut a = Array::from_shape_fn((5, 10).set_f(config.a_is_f()), |idx| idx);
        let mut b = Array::from_shape_fn((5, 10).set_f(config.b_is_f()), |_| 0);
        if slice {
            a = a.slice_move(s![.., ..-1]);
            b = b.slice_move(s![.., ..-1]);
        }

        let _result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            Zip::from(&a).and(&b).map_collect(|&elt, _| {
                if elt.0 > 3 && will_panic {
                    panic!();
                }
                inserts.borrow_mut().push(elt);
                Recorddrop(elt, &drops)
            });
        }));

        println!("{:?}", inserts.get_mut());
        println!("{:?}", drops.get_mut());

        assert_eq!(inserts.get_mut().len(), drops.get_mut().len(), "Incorrect number of drops");
        assert_eq!(inserts.get_mut(), drops.get_mut(), "Incorrect order of drops");
    };

    for &should_panic in &[true, false] {
        for &should_slice in &[false, true] {
            test_collect_panic(Config::CC, should_panic, should_slice);
            test_collect_panic(Config::CF, should_panic, should_slice);
            test_collect_panic(Config::FF, should_panic, should_slice);
        }
    }
}


#[test]
fn test_azip_syntax_trailing_comma() {
    let mut b = Array::<i32, _>::zeros((5, 5));
    let mut c = Array::<i32, _>::ones((5, 5));
    let a = b.clone();
    azip!((b in &mut b, c in &mut c, ) swap(b, c));
    assert_eq!(a, c);
    assert!(a != b);
}

#[test]
#[cfg(feature = "approx")]
fn test_azip2_sum() {
    use approx::assert_abs_diff_eq;

    let c = Array::from_shape_fn((5, 10), |(i, j)| f32::exp((i + j) as f32));
    for i in 0..2 {
        let ax = Axis(i);
        let mut b = Array::zeros(c.len_of(ax));
        azip!((b in &mut b, c in c.axis_iter(ax)) *b = c.sum());
        assert_abs_diff_eq!(b, c.sum_axis(Axis(1 - i)), epsilon = 1e-6);
    }
}

#[test]
#[cfg(feature = "approx")]
fn test_azip3_slices() {
    use approx::assert_abs_diff_eq;

    let mut a = [0.; 32];
    let mut b = [0.; 32];
    let mut c = [0.; 32];
    for (i, elt) in b.iter_mut().enumerate() {
        *elt = i as f32;
    }

    azip!((a in &mut a[..], b in &b[..], c in &mut c[..]) {
        *a += b / 10.;
        *c = a.sin();
    });
    let res = Array::linspace(0., 3.1, 32).mapv_into(f32::sin);
    assert_abs_diff_eq!(res, ArrayView::from(&c), epsilon = 1e-4);
}

#[test]
#[cfg(feature = "approx")]
fn test_broadcast() {
    use approx::assert_abs_diff_eq;

    let n = 16;
    let mut a = Array::<f32, _>::zeros((n, n));
    let mut b = Array::<f32, _>::from_elem((1, n), 1.);
    for ((i, j), elt) in b.indexed_iter_mut() {
        *elt /= 1. + (i + 2 * j) as f32;
    }
    let d = Array::from_elem((1, n), 1.);
    let e = Array::from_elem((), 2.);

    {
        let z = Zip::from(a.view_mut())
            .and_broadcast(&b)
            .and_broadcast(&d)
            .and_broadcast(&e);
        z.for_each(|x, &y, &z, &w| *x = y + z + w);
    }
    let sum = &b + &d + &e;
    assert_abs_diff_eq!(a, sum.broadcast((n, n)).unwrap(), epsilon = 1e-4);
}

#[should_panic]
#[test]
fn test_zip_dim_mismatch_1() {
    let mut a = Array::zeros((5, 7));
    let mut d = a.raw_dim();
    d[0] += 1;
    let b = Array::from_shape_fn(d, |(i, j)| 1. / (i + 2 * j) as f32);
    azip!((a in &mut a, &b in &b) *a = b);
}

// Test that Zip handles memory layout correctly for
// Zip::from(A).and(B)
// where A is F-contiguous and B contiguous but neither F nor C contiguous.
#[test]
fn test_contiguous_but_not_c_or_f() {
    let a = Array::from_iter(0..27).into_shape((3, 3, 3)).unwrap();

    // both F order
    let a = a.reversed_axes();
    let mut b = a.clone();
    assert_eq!(a.strides(), b.strides());
    assert_eq!(a.strides(), &[1, 3, 9]);
    b.swap_axes(0, 1);
    // test single elem so that test keeps working if array `+` impl changes
    let correct = &a + &b;
    let correct_012 = a[[0, 1, 2]] + b[[0, 1, 2]];

    let mut ans = Array::zeros(a.dim().f());
    azip!((ans in &mut ans, &a in &a, &b in &b) *ans = a + b);
    println!("{:?}", a);
    println!("{:?}", b);
    println!("{:?}", ans);

    assert_eq!(ans[[0, 1, 2]], correct_012);
    assert_eq!(ans, correct);
}

#[test]
fn test_clone() {
    let a = Array::from_iter(0..27).into_shape((3, 3, 3)).unwrap();

    let z = Zip::from(&a).and(a.exact_chunks((1, 1, 1)));
    let w = z.clone();
    let mut result = Vec::new();
    z.for_each(|x, y| {
        result.push((x, y));
    });
    let mut i = 0;
    w.for_each(|x, y| {
        assert_eq!(result[i], (x, y));
        i += 1;
    });
}

#[test]
fn test_indices_0() {
    let a1 = arr0(3);

    let mut count = 0;
    Zip::indexed(&a1).for_each(|i, elt| {
        count += 1;
        assert_eq!(i, ());
        assert_eq!(*elt, 3);
    });
    assert_eq!(count, 1);
}

#[test]
fn test_indices_1() {
    let mut a1 = Array::default(12);
    for (i, elt) in a1.indexed_iter_mut() {
        *elt = i;
    }

    let mut count = 0;
    Zip::indexed(&a1).for_each(|i, elt| {
        count += 1;
        assert_eq!(*elt, i);
    });
    assert_eq!(count, a1.len());

    let mut count = 0;
    let len = a1.len();
    let (x, y) = Zip::indexed(&mut a1).split();

    x.for_each(|i, elt| {
        count += 1;
        assert_eq!(*elt, i);
    });
    assert_eq!(count, len / 2);
    y.for_each(|i, elt| {
        count += 1;
        assert_eq!(*elt, i);
    });
    assert_eq!(count, len);
}

#[test]
fn test_indices_2() {
    let mut a1 = Array::default((10, 12));
    for (i, elt) in a1.indexed_iter_mut() {
        *elt = i;
    }

    let mut count = 0;
    azip!((index i, &a1 in &a1) {
        count += 1;
        assert_eq!(a1, i);
    });
    assert_eq!(count, a1.len());

    let mut count = 0;
    let len = a1.len();
    let (x, y) = Zip::indexed(&mut a1).split();

    x.for_each(|i, elt| {
        count += 1;
        assert_eq!(*elt, i);
    });
    assert_eq!(count, len / 2);
    y.for_each(|i, elt| {
        count += 1;
        assert_eq!(*elt, i);
    });
    assert_eq!(count, len);
}

#[test]
fn test_indices_3() {
    let mut a1 = Array::default((4, 5, 6));
    for (i, elt) in a1.indexed_iter_mut() {
        *elt = i;
    }

    let mut count = 0;
    Zip::indexed(&a1).for_each(|i, elt| {
        count += 1;
        assert_eq!(*elt, i);
    });
    assert_eq!(count, a1.len());

    let mut count = 0;
    let len = a1.len();
    let (x, y) = Zip::indexed(&mut a1).split();

    x.for_each(|i, elt| {
        count += 1;
        assert_eq!(*elt, i);
    });
    assert_eq!(count, len / 2);
    y.for_each(|i, elt| {
        count += 1;
        assert_eq!(*elt, i);
    });
    assert_eq!(count, len);
}

#[test]
fn test_indices_split_1() {
    for m in (0..4).chain(10..12) {
        for n in (0..4).chain(10..12) {
            let a1 = Array::<f64, _>::default((m, n));
            if a1.len() <= 1 {
                continue;
            }
            let (a, b) = Zip::indexed(&a1).split();
            let mut seen = Vec::new();

            let mut ac = 0;
            a.for_each(|i, _| {
                ac += 1;
                seen.push(i);
            });
            let mut bc = 0;
            b.for_each(|i, _| {
                bc += 1;
                seen.push(i);
            });

            assert_eq!(a1.len(), ac + bc);

            seen.sort();
            assert_eq!(seen.len(), a1.len());
            seen.dedup();
            assert_eq!(seen.len(), a1.len());
        }
    }
}

#[test]
fn test_zip_all() {
    let a = Array::<f32, _>::zeros(62);
    let b = Array::<f32, _>::ones(62);
    let mut c = Array::<f32, _>::ones(62);
    c[5] = 0.0;
    assert_eq!(true, Zip::from(&a).and(&b).all(|&x, &y| x + y == 1.0));
    assert_eq!(false, Zip::from(&a).and(&b).all(|&x, &y| x == y));
    assert_eq!(false, Zip::from(&a).and(&c).all(|&x, &y| x + y == 1.0));
}

#[test]
fn test_zip_all_empty_array() {
    let a = Array::<f32, _>::zeros(0);
    let b = Array::<f32, _>::ones(0);
    assert_eq!(true, Zip::from(&a).and(&b).all(|&_x, &_y| true));
    assert_eq!(true, Zip::from(&a).and(&b).all(|&_x, &_y| false));
}
