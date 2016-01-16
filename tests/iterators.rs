extern crate ndarray;
extern crate itertools;

use ndarray::Array;
use ndarray::{Ix, Si, S};
use ndarray::{
    ArrayBase,
    Data,
    Dimension,
    aview1,
    arr3,
};

use itertools::assert_equal;
use itertools::{rev, enumerate};

#[test]
fn double_ended() {
    let a = Array::linspace(0., 7., 8);
    let mut it = a.iter().map(|x| *x);
    assert_eq!(it.next(), Some(0.));
    assert_eq!(it.next_back(), Some(7.));
    assert_eq!(it.next(), Some(1.));
    assert_eq!(it.rev().last(), Some(2.));
    assert_equal(aview1(&[1, 2, 3]), &[1, 2, 3]);
    assert_equal(rev(aview1(&[1, 2, 3])), rev(&[1, 2, 3]));
}

#[test]
fn iter_size_hint() {
    // Check that the size hint is correctly computed
    let a = Array::from_iter(0..24).reshape((2, 3, 4));
    let mut data = [0; 24];
    for (i, elt) in enumerate(&mut data) {
        *elt = i as i32;
    }
    assert_equal(&a, &data);
    let mut it = a.iter();
    let mut ans = data.iter();
    assert_eq!(it.len(), ans.len());
    while ans.len() > 0 {
        assert_eq!(it.next(), ans.next());
        assert_eq!(it.len(), ans.len());
    }
}

#[test]
fn indexed()
{
    let a = Array::linspace(0., 7., 8);
    for (i, elt) in a.indexed_iter() {
        assert_eq!(i, *elt as Ix);
    }
    let a = a.reshape((2, 4, 1));
    let (mut i, mut j, k) = (0, 0, 0);
    for (idx, elt) in a.indexed_iter() {
        assert_eq!(idx, (i, j, k));
        j += 1;
        if j == 4 {
            j = 0;
            i += 1;
        }
        println!("{:?}", (idx, elt));
    }
}


fn assert_slice_correct<A, S, D>(v: &ArrayBase<S, D>)
    where S: Data<Elem=A>,
          D: Dimension,
          A: PartialEq + std::fmt::Debug,
{
    let slc = v.as_slice();
    assert!(slc.is_some());
    let slc = slc.unwrap();
    assert_eq!(v.len(), slc.len());
    assert_equal(v.iter(), slc);
}

#[test]
fn as_slice() {
    let a = Array::linspace(0., 7., 8);
    let a = a.reshape((2, 4, 1));

    assert_slice_correct(&a);

    let a = a.reshape((2, 4));
    assert_slice_correct(&a);

    assert!(a.view().subview(1, 0).as_slice().is_none());

    let v = a.view();
    assert_slice_correct(&v);
    assert_slice_correct(&v.subview(0, 0));
    assert_slice_correct(&v.subview(0, 1));

    assert!(v.slice(&[S, Si(0, Some(1), 1)]).as_slice().is_none());
    println!("{:?}", v.slice(&[Si(0, Some(1), 2), S]));
    assert!(v.slice(&[Si(0, Some(1), 2), S]).as_slice().is_some());

    // `u` is contiguous, because the column stride of `2` doesn't matter
    // when the result is just one row anyway -- length of that dimension is 1
    let u = v.slice(&[Si(0, Some(1), 2), S]);
    println!("{:?}", u.shape());
    println!("{:?}", u.strides());
    println!("{:?}", v.slice(&[Si(0, Some(1), 2), S]));
    assert!(u.as_slice().is_some());
    assert_slice_correct(&u);

    let a = a.reshape((8, 1));
    assert_slice_correct(&a);
    let u = a.slice(&[Si(0, None, 2), S]);
    println!("u={:?}, shape={:?}, strides={:?}", u, u.shape(), u.strides());
    assert!(u.as_slice().is_none());
}

#[test]
fn inner_iter() {
    let a = Array::from_iter(0..12);
    let a = a.reshape((2, 3, 2));
    // [[[0, 1],
    //   [2, 3],
    //   [4, 5]],
    //  [[6, 7],
    //   [8, 9],
    //    ...
    assert_equal(a.inner_iter(),
                 vec![aview1(&[0, 1]), aview1(&[2, 3]), aview1(&[4, 5]),
                      aview1(&[6, 7]), aview1(&[8, 9]), aview1(&[10, 11])]);
    let mut b = Array::zeros((2, 3, 2));
    b.swap_axes(0, 2);
    b.assign(&a);
    assert_equal(b.inner_iter(),
                 vec![aview1(&[0, 1]), aview1(&[2, 3]), aview1(&[4, 5]),
                      aview1(&[6, 7]), aview1(&[8, 9]), aview1(&[10, 11])]);
}

#[test]
fn inner_iter_corner_cases() {
    let a0 = Array::zeros(());
    assert_equal(a0.inner_iter(), vec![aview1(&[0])]);

    let a2 = Array::<i32, _>::zeros((0, 3));
    assert_equal(a2.inner_iter(),
                 vec![aview1(&[]); 0]);

    let a2 = Array::<i32, _>::zeros((3, 0));
    assert_equal(a2.inner_iter(),
                 vec![aview1(&[]); 3]);
}

#[test]
fn inner_iter_size_hint() {
    // Check that the size hint is correctly computed
    let a = Array::from_iter(0..24).reshape((2, 3, 4));
    let mut len = 6;
    let mut it = a.inner_iter();
    assert_eq!(it.len(), len);
    while len > 0 {
        it.next();
        len -= 1;
        assert_eq!(it.len(), len);
    }
}

#[test]
fn outer_iter() {
    let a = Array::from_iter(0..12);
    let a = a.reshape((2, 3, 2));
    // [[[0, 1],
    //   [2, 3],
    //   [4, 5]],
    //  [[6, 7],
    //   [8, 9],
    //    ...
    assert_equal(a.outer_iter(),
                 vec![a.subview(0, 0), a.subview(0, 1)]);
    let mut b = Array::zeros((2, 3, 2));
    b.swap_axes(0, 2);
    b.assign(&a);
    assert_equal(b.outer_iter(),
                 vec![a.subview(0, 0), a.subview(0, 1)]);

    let mut found_rows = Vec::new();
    for sub in b.outer_iter() {
        for row in sub.into_outer_iter() {
            found_rows.push(row);
        }
    }
    assert_equal(a.inner_iter(), found_rows.clone());

    let mut found_rows_rev = Vec::new();
    for sub in b.outer_iter().rev() {
        for row in sub.into_outer_iter().rev() {
            found_rows_rev.push(row);
        }
    }
    found_rows_rev.reverse();
    assert_eq!(&found_rows, &found_rows_rev);
}

#[test]
fn axis_iter() {
    let a = Array::from_iter(0..12);
    let a = a.reshape((2, 3, 2));
    // [[[0, 1],
    //   [2, 3],
    //   [4, 5]],
    //  [[6, 7],
    //   [8, 9],
    //    ...
    assert_equal(a.axis_iter(1),
                 vec![a.subview(1, 0),
                      a.subview(1, 1),
                      a.subview(1, 2)]);
}

#[test]
fn outer_iter_corner_cases() {
    let a2 = Array::<i32, _>::zeros((0, 3));
    assert_equal(a2.outer_iter(),
                 vec![aview1(&[]); 0]);

    let a2 = Array::<i32, _>::zeros((3, 0));
    assert_equal(a2.outer_iter(),
                 vec![aview1(&[]); 3]);
}

#[test]
fn outer_iter_mut() {
    let a = Array::from_iter(0..12);
    let a = a.reshape((2, 3, 2));
    // [[[0, 1],
    //   [2, 3],
    //   [4, 5]],
    //  [[6, 7],
    //   [8, 9],
    //    ...
    let mut b = Array::zeros((2, 3, 2));
    b.swap_axes(0, 2);
    b.assign(&a);
    assert_equal(b.outer_iter_mut(),
                 vec![a.subview(0, 0), a.subview(0, 1)]);

    let mut found_rows = Vec::new();
    for sub in b.outer_iter_mut() {
        for row in sub.into_outer_iter() {
            found_rows.push(row);
        }
    }
    assert_equal(a.inner_iter(), found_rows);
}

#[test]
fn axis_iter_mut() {
    let a = Array::from_iter(0..12);
    let a = a.reshape((2, 3, 2));
    // [[[0, 1],
    //   [2, 3],
    //   [4, 5]],
    //  [[6, 7],
    //   [8, 9],
    //    ...
    let mut a = a.to_owned();

    for mut subview in a.axis_iter_mut(1) {
        subview[[0, 0]] = 42;
    }

    let b = arr3(&[[[42, 1],
                    [42, 3],
                    [42, 5]],
                   [[6, 7],
                    [8, 9],
                    [10, 11]]]);
    assert_eq!(a, b);
}

#[test]
fn outer_iter_size_hint() {
    // Check that the size hint is correctly computed
    let a = Array::from_iter(0..24).reshape((4, 3, 2));
    let mut len = 4;
    let mut it = a.outer_iter();
    assert_eq!(it.len(), len);
    while len > 0 {
        it.next();
        len -= 1;
        assert_eq!(it.len(), len);
    }

    // now try the double ended case
    let mut it = a.outer_iter();
    it.next_back();
    let mut len = 3;
    while len > 0 {
        it.next();
        len -= 1;
        assert_eq!(it.len(), len);
    }

    let mut it = a.outer_iter();
    it.next();
    let mut len = 3;
    while len > 0 {
        it.next_back();
        len -= 1;
        assert_eq!(it.len(), len);
    }
}
