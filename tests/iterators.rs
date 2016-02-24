#[macro_use(s)]
extern crate ndarray;
extern crate itertools;

use ndarray::RcArray;
use ndarray::{Ix, Si, S};
use ndarray::{
    ArrayBase,
    Data,
    Dimension,
    aview1,
    arr2,
    arr3,
};

use itertools::assert_equal;
use itertools::{rev, enumerate};

#[test]
fn double_ended() {
    let a = RcArray::linspace(0., 7., 8);
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
    let a = RcArray::from_iter(0..24).reshape((2, 3, 4));
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
    let a = RcArray::linspace(0., 7., 8);
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
    let a = RcArray::linspace(0., 7., 8);
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
    let a = RcArray::from_iter(0..12);
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
    let mut b = RcArray::zeros((2, 3, 2));
    b.swap_axes(0, 2);
    b.assign(&a);
    assert_equal(b.inner_iter(),
                 vec![aview1(&[0, 1]), aview1(&[2, 3]), aview1(&[4, 5]),
                      aview1(&[6, 7]), aview1(&[8, 9]), aview1(&[10, 11])]);
}

#[test]
fn inner_iter_corner_cases() {
    let a0 = RcArray::zeros(());
    assert_equal(a0.inner_iter(), vec![aview1(&[0])]);

    let a2 = RcArray::<i32, _>::zeros((0, 3));
    assert_equal(a2.inner_iter(),
                 vec![aview1(&[]); 0]);

    let a2 = RcArray::<i32, _>::zeros((3, 0));
    assert_equal(a2.inner_iter(),
                 vec![aview1(&[]); 3]);
}

#[test]
fn inner_iter_size_hint() {
    // Check that the size hint is correctly computed
    let a = RcArray::from_iter(0..24).reshape((2, 3, 4));
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
    let a = RcArray::from_iter(0..12);
    let a = a.reshape((2, 3, 2));
    // [[[0, 1],
    //   [2, 3],
    //   [4, 5]],
    //  [[6, 7],
    //   [8, 9],
    //    ...
    assert_equal(a.outer_iter(),
                 vec![a.subview(0, 0), a.subview(0, 1)]);
    let mut b = RcArray::zeros((2, 3, 2));
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

    // Test a case where strides are negative instead
    let mut c = RcArray::zeros((2, 3, 2));
    let mut cv = c.slice_mut(s![..;-1, ..;-1, ..;-1]);
    cv.assign(&a);
    assert_eq!(&a, &cv);
    assert_equal(cv.outer_iter(),
                 vec![a.subview(0, 0), a.subview(0, 1)]);

    let mut found_rows = Vec::new();
    for sub in cv.outer_iter() {
        for row in sub.into_outer_iter() {
            found_rows.push(row);
        }
    }
    println!("{:#?}", found_rows);
    assert_equal(a.inner_iter(), found_rows);
}

#[test]
fn axis_iter() {
    let a = RcArray::from_iter(0..12);
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
    let a2 = RcArray::<i32, _>::zeros((0, 3));
    assert_equal(a2.outer_iter(),
                 vec![aview1(&[]); 0]);

    let a2 = RcArray::<i32, _>::zeros((3, 0));
    assert_equal(a2.outer_iter(),
                 vec![aview1(&[]); 3]);
}

#[test]
fn outer_iter_mut() {
    let a = RcArray::from_iter(0..12);
    let a = a.reshape((2, 3, 2));
    // [[[0, 1],
    //   [2, 3],
    //   [4, 5]],
    //  [[6, 7],
    //   [8, 9],
    //    ...
    let mut b = RcArray::zeros((2, 3, 2));
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
    let a = RcArray::from_iter(0..12);
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
fn axis_chunks_iter() {
    let a = RcArray::from_iter(0..24);
    let a = a.reshape((2, 6, 2));

    let it = a.axis_chunks_iter(1, 2);
    assert_equal(it,
                 vec![arr3(&[[[0, 1], [2, 3]], [[12, 13], [14, 15]]]),
                      arr3(&[[[4, 5], [6, 7]], [[16, 17], [18, 19]]]),
                      arr3(&[[[8, 9], [10, 11]], [[20, 21], [22, 23]]])]);

    let a = RcArray::from_iter(0..28);
    let a = a.reshape((2, 7, 2));

    let it = a.axis_chunks_iter(1, 2);
    assert_equal(it,
                 vec![arr3(&[[[0, 1], [2, 3]], [[14, 15], [16, 17]]]),
                      arr3(&[[[4, 5], [6, 7]], [[18, 19], [20, 21]]]),
                      arr3(&[[[8, 9], [10, 11]], [[22, 23], [24, 25]]]),
                      arr3(&[[[12, 13]], [[26, 27]]])]);

    let it = a.axis_chunks_iter(1, 2).rev();
    assert_equal(it,
                 vec![arr3(&[[[12, 13]], [[26, 27]]]),
                      arr3(&[[[8, 9], [10, 11]], [[22, 23], [24, 25]]]),
                      arr3(&[[[4, 5], [6, 7]], [[18, 19], [20, 21]]]),
                      arr3(&[[[0, 1], [2, 3]], [[14, 15], [16, 17]]])]);

    let it = a.axis_chunks_iter(1, 7);
    assert_equal(it, vec![a.view()]);

    let it = a.axis_chunks_iter(1, 9);
    assert_equal(it, vec![a.view()]);
}

#[test]
fn axis_chunks_iter_corner_cases() {
    // examples provided by @bluss in PR #65
    // these tests highlight corner cases of the axis_chunks_iter implementation
    // and enable checking if no pointer offseting is out of bounds. However
    // checking the absence of of out of bounds offseting cannot (?) be
    // done automatically, so one has to launch this test in a debugger.
    let a = RcArray::<f32, _>::linspace(0., 7., 8).reshape((8, 1));
    let it = a.axis_chunks_iter(0, 4);
    assert_equal(it, vec![a.slice(s![..4, ..]), a.slice(s![4.., ..])]);
    let a = a.slice(s![..;-1,..]);
    let it = a.axis_chunks_iter(0, 8);
    assert_equal(it, vec![a.view()]);
    let it = a.axis_chunks_iter(0, 3);
    assert_equal(it,
                 vec![arr2(&[[7.], [6.], [5.]]),
                      arr2(&[[4.], [3.], [2.]]),
                      arr2(&[[1.], [0.]])]);

    let b = RcArray::<f32, _>::zeros((8, 2));
    let a = b.slice(s![1..;2,..]);
    let it = a.axis_chunks_iter(0, 8);
    assert_equal(it, vec![a.view()]);

    let it = a.axis_chunks_iter(0, 1);
    assert_equal(it, vec![RcArray::zeros((1, 2)); 4]);
}

#[test]
fn axis_chunks_iter_mut() {
    let a = RcArray::from_iter(0..24);
    let mut a = a.reshape((2, 6, 2));

    let mut it = a.axis_chunks_iter_mut(1, 2);
    let mut col0 = it.next().unwrap();
    col0[[0, 0, 0]] = 42;
    assert_eq!(col0, arr3(&[[[42, 1], [2, 3]], [[12, 13], [14, 15]]]));
}

#[test]
fn outer_iter_size_hint() {
    // Check that the size hint is correctly computed
    let a = RcArray::from_iter(0..24).reshape((4, 3, 2));
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

#[test]
fn outer_iter_split_at() {
    let a = Array::from_iter(0..30).reshape((5, 3, 2));

    let it = a.outer_iter();
    let (mut itl, mut itr) = it.clone().split_at(2);
    assert_eq!(itl.next().unwrap()[[2, 1]], 5);
    assert_eq!(itl.next().unwrap()[[2, 1]], 11);
    assert_eq!(itl.next(), None);

    assert_eq!(itr.next().unwrap()[[2, 1]], 17);
    assert_eq!(itr.next().unwrap()[[2, 1]], 23);
    assert_eq!(itr.next().unwrap()[[2, 1]], 29);
    assert_eq!(itr.next(), None);

    // split_at on length should yield an empty iterator
    // on the right part
    let (_, mut itr) = it.split_at(5);
    assert_eq!(itr.next(), None);
}

#[test]
#[should_panic]
fn outer_iter_split_at_panics() {
    let a = Array::from_iter(0..30).reshape((5, 3, 2));

    let it = a.outer_iter();
    it.split_at(6);
}

#[test]
fn outer_iter_mut_split_at() {
    let mut a = Array::from_iter(0..30).reshape((5, 3, 2));

    {
        let it = a.outer_iter_mut();
        let (mut itl, mut itr) = it.split_at(2);
        itl.next();
        itl.next().unwrap()[[2, 1]] += 1; // now this value is 12
        assert_eq!(itl.next(), None);

        itr.next();
        itr.next();
        itr.next().unwrap()[[2, 1]] -= 1; // now this value is 28
        assert_eq!(itr.next(), None);
    }
    assert_eq!(a[[1, 2, 1]], 12);
    assert_eq!(a[[4, 2, 1]], 28);
}
