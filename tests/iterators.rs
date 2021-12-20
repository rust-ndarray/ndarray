#![allow(
    clippy::many_single_char_names,
    clippy::deref_addrof,
    clippy::unreadable_literal,
    clippy::many_single_char_names
)]

use ndarray::prelude::*;
use ndarray::{arr3, aview1, indices, s, Axis, Slice, Zip};

use itertools::assert_equal;
use itertools::enumerate;
use std::cell::Cell;

macro_rules! assert_panics {
    ($body:expr) => {
        if let Ok(v) = ::std::panic::catch_unwind(|| $body) {
            panic!("assertion failed: should_panic; \
            non-panicking result: {:?}", v);
        }
    };
    ($body:expr, $($arg:tt)*) => {
        if let Ok(_) = ::std::panic::catch_unwind(|| $body) {
            panic!($($arg)*);
        }
    };
}

#[test]
#[cfg(feature = "std")]
fn double_ended() {
    let a = ArcArray::linspace(0., 7., 8);
    let mut it = a.iter().cloned();
    assert_eq!(it.next(), Some(0.));
    assert_eq!(it.next_back(), Some(7.));
    assert_eq!(it.next(), Some(1.));
    assert_eq!(it.rev().last(), Some(2.));
    assert_equal(aview1(&[1, 2, 3]), &[1, 2, 3]);
    assert_equal(aview1(&[1, 2, 3]).into_iter().rev(), [1, 2, 3].iter().rev());
}

#[test]
fn iter_size_hint() {
    // Check that the size hint is correctly computed
    let a = ArcArray::from_iter(0..24).reshape((2, 3, 4));
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
#[cfg(feature = "std")]
fn indexed() {
    let a = ArcArray::linspace(0., 7., 8);
    for (i, elt) in a.indexed_iter() {
        assert_eq!(i, *elt as usize);
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

#[test]
#[cfg(feature = "std")]
fn as_slice() {
    use ndarray::Data;

    fn assert_slice_correct<A, S, D>(v: &ArrayBase<S, D>)
    where
        S: Data<Elem = A>,
        D: Dimension,
        A: PartialEq + std::fmt::Debug,
    {
        let slc = v.as_slice();
        assert!(slc.is_some());
        let slc = slc.unwrap();
        assert_eq!(v.len(), slc.len());
        assert_equal(v.iter(), slc);
    }

    let a = ArcArray::linspace(0., 7., 8);
    let a = a.reshape((2, 4, 1));

    assert_slice_correct(&a);

    let a = a.reshape((2, 4));
    assert_slice_correct(&a);

    assert!(a.view().index_axis(Axis(1), 0).as_slice().is_none());

    let v = a.view();
    assert_slice_correct(&v);
    assert_slice_correct(&v.index_axis(Axis(0), 0));
    assert_slice_correct(&v.index_axis(Axis(0), 1));

    assert!(v.slice(s![.., ..1]).as_slice().is_none());
    println!("{:?}", v.slice(s![..1;2, ..]));
    assert!(v.slice(s![..1;2, ..]).as_slice().is_some());

    // `u` is contiguous, because the column stride of `2` doesn't matter
    // when the result is just one row anyway -- length of that dimension is 1
    let u = v.slice(s![..1;2, ..]);
    println!("{:?}", u.shape());
    println!("{:?}", u.strides());
    println!("{:?}", v.slice(s![..1;2, ..]));
    assert!(u.as_slice().is_some());
    assert_slice_correct(&u);

    let a = a.reshape((8, 1));
    assert_slice_correct(&a);
    let u = a.slice(s![..;2, ..]);
    println!(
        "u={:?}, shape={:?}, strides={:?}",
        u,
        u.shape(),
        u.strides()
    );
    assert!(u.as_slice().is_none());
}

#[test]
fn inner_iter() {
    let a = ArcArray::from_iter(0..12);
    let a = a.reshape((2, 3, 2));
    // [[[0, 1],
    //   [2, 3],
    //   [4, 5]],
    //  [[6, 7],
    //   [8, 9],
    //    ...
    assert_equal(
        a.rows(),
        vec![
            aview1(&[0, 1]),
            aview1(&[2, 3]),
            aview1(&[4, 5]),
            aview1(&[6, 7]),
            aview1(&[8, 9]),
            aview1(&[10, 11]),
        ],
    );
    let mut b = ArcArray::zeros((2, 3, 2));
    b.swap_axes(0, 2);
    b.assign(&a);
    assert_equal(
        b.rows(),
        vec![
            aview1(&[0, 1]),
            aview1(&[2, 3]),
            aview1(&[4, 5]),
            aview1(&[6, 7]),
            aview1(&[8, 9]),
            aview1(&[10, 11]),
        ],
    );
}

#[test]
fn inner_iter_corner_cases() {
    let a0 = ArcArray::<i32, _>::zeros(());
    assert_equal(a0.rows(), vec![aview1(&[0])]);

    let a2 = ArcArray::<i32, _>::zeros((0, 3));
    assert_equal(a2.rows(), vec![aview1(&[]); 0]);

    let a2 = ArcArray::<i32, _>::zeros((3, 0));
    assert_equal(a2.rows(), vec![aview1(&[]); 3]);
}

#[test]
fn inner_iter_size_hint() {
    // Check that the size hint is correctly computed
    let a = ArcArray::from_iter(0..24).reshape((2, 3, 4));
    let mut len = 6;
    let mut it = a.rows().into_iter();
    assert_eq!(it.len(), len);
    while len > 0 {
        it.next();
        len -= 1;
        assert_eq!(it.len(), len);
    }
}

#[allow(deprecated)] // into_outer_iter
#[test]
fn outer_iter() {
    let a = ArcArray::from_iter(0..12);
    let a = a.reshape((2, 3, 2));
    // [[[0, 1],
    //   [2, 3],
    //   [4, 5]],
    //  [[6, 7],
    //   [8, 9],
    //    ...
    assert_equal(
        a.outer_iter(),
        vec![a.index_axis(Axis(0), 0), a.index_axis(Axis(0), 1)],
    );
    let mut b = ArcArray::zeros((2, 3, 2));
    b.swap_axes(0, 2);
    b.assign(&a);
    assert_equal(
        b.outer_iter(),
        vec![a.index_axis(Axis(0), 0), a.index_axis(Axis(0), 1)],
    );

    let mut found_rows = Vec::new();
    for sub in b.outer_iter() {
        for row in sub.into_outer_iter() {
            found_rows.push(row);
        }
    }
    assert_equal(a.rows(), found_rows.clone());

    let mut found_rows_rev = Vec::new();
    for sub in b.outer_iter().rev() {
        for row in sub.into_outer_iter().rev() {
            found_rows_rev.push(row);
        }
    }
    found_rows_rev.reverse();
    assert_eq!(&found_rows, &found_rows_rev);

    // Test a case where strides are negative instead
    let mut c = ArcArray::zeros((2, 3, 2));
    let mut cv = c.slice_mut(s![..;-1, ..;-1, ..;-1]);
    cv.assign(&a);
    assert_eq!(&a, &cv);
    assert_equal(
        cv.outer_iter(),
        vec![a.index_axis(Axis(0), 0), a.index_axis(Axis(0), 1)],
    );

    let mut found_rows = Vec::new();
    for sub in cv.outer_iter() {
        for row in sub.into_outer_iter() {
            found_rows.push(row);
        }
    }
    println!("{:#?}", found_rows);
    assert_equal(a.rows(), found_rows);
}

#[test]
fn axis_iter() {
    let a = ArcArray::from_iter(0..12);
    let a = a.reshape((2, 3, 2));
    // [[[0, 1],
    //   [2, 3],
    //   [4, 5]],
    //  [[6, 7],
    //   [8, 9],
    //    ...
    assert_equal(
        a.axis_iter(Axis(1)),
        vec![
            a.index_axis(Axis(1), 0),
            a.index_axis(Axis(1), 1),
            a.index_axis(Axis(1), 2),
        ],
    );
}

#[test]
fn axis_iter_split_at() {
    let a = Array::from_iter(0..5);
    let iter = a.axis_iter(Axis(0));
    let all: Vec<_> = iter.clone().collect();
    for mid in 0..=all.len() {
        let (left, right) = iter.clone().split_at(mid);
        assert_eq!(&all[..mid], &left.collect::<Vec<_>>()[..]);
        assert_eq!(&all[mid..], &right.collect::<Vec<_>>()[..]);
    }
}

#[test]
fn axis_iter_split_at_partially_consumed() {
    let a = Array::from_iter(0..5);
    let mut iter = a.axis_iter(Axis(0));
    while iter.next().is_some() {
        let remaining: Vec<_> = iter.clone().collect();
        for mid in 0..=remaining.len() {
            let (left, right) = iter.clone().split_at(mid);
            assert_eq!(&remaining[..mid], &left.collect::<Vec<_>>()[..]);
            assert_eq!(&remaining[mid..], &right.collect::<Vec<_>>()[..]);
        }
    }
}

#[test]
fn axis_iter_zip() {
    let a = Array::from_iter(0..5);
    let iter = a.axis_iter(Axis(0));
    let mut b = Array::zeros(5);
    Zip::from(&mut b).and(iter).for_each(|b, a| *b = a[()]);
    assert_eq!(a, b);
}

#[test]
fn axis_iter_zip_partially_consumed() {
    let a = Array::from_iter(0..5);
    let mut iter = a.axis_iter(Axis(0));
    let mut consumed = 0;
    while iter.next().is_some() {
        consumed += 1;
        let mut b = Array::zeros(a.len() - consumed);
        Zip::from(&mut b).and(iter.clone()).for_each(|b, a| *b = a[()]);
        assert_eq!(a.slice(s![consumed..]), b);
    }
}

#[test]
fn axis_iter_zip_partially_consumed_discontiguous() {
    let a = Array::from_iter(0..5);
    let mut iter = a.axis_iter(Axis(0));
    let mut consumed = 0;
    while iter.next().is_some() {
        consumed += 1;
        let mut b = Array::zeros((a.len() - consumed) * 2);
        b.slice_collapse(s![..;2]);
        Zip::from(&mut b).and(iter.clone()).for_each(|b, a| *b = a[()]);
        assert_eq!(a.slice(s![consumed..]), b);
    }
}

#[test]
fn outer_iter_corner_cases() {
    let a2 = ArcArray::<i32, _>::zeros((0, 3));
    assert_equal(a2.outer_iter(), vec![aview1(&[]); 0]);

    let a2 = ArcArray::<i32, _>::zeros((3, 0));
    assert_equal(a2.outer_iter(), vec![aview1(&[]); 3]);
}

#[allow(deprecated)]
#[test]
fn outer_iter_mut() {
    let a = ArcArray::from_iter(0..12);
    let a = a.reshape((2, 3, 2));
    // [[[0, 1],
    //   [2, 3],
    //   [4, 5]],
    //  [[6, 7],
    //   [8, 9],
    //    ...
    let mut b = ArcArray::zeros((2, 3, 2));
    b.swap_axes(0, 2);
    b.assign(&a);
    assert_equal(
        b.outer_iter_mut(),
        vec![a.index_axis(Axis(0), 0), a.index_axis(Axis(0), 1)],
    );

    let mut found_rows = Vec::new();
    for sub in b.outer_iter_mut() {
        for row in sub.into_outer_iter() {
            found_rows.push(row);
        }
    }
    assert_equal(a.rows(), found_rows);
}

#[test]
fn axis_iter_mut() {
    let a = ArcArray::from_iter(0..12);
    let a = a.reshape((2, 3, 2));
    // [[[0, 1],
    //   [2, 3],
    //   [4, 5]],
    //  [[6, 7],
    //   [8, 9],
    //    ...
    let mut a = a.to_owned();

    for mut subview in a.axis_iter_mut(Axis(1)) {
        subview[[0, 0]] = 42;
    }

    let b = arr3(&[[[42, 1], [42, 3], [42, 5]], [[6, 7], [8, 9], [10, 11]]]);
    assert_eq!(a, b);
}

#[test]
fn axis_chunks_iter() {
    let a = ArcArray::from_iter(0..24);
    let a = a.reshape((2, 6, 2));

    let it = a.axis_chunks_iter(Axis(1), 2);
    assert_equal(
        it,
        vec![
            arr3(&[[[0, 1], [2, 3]], [[12, 13], [14, 15]]]),
            arr3(&[[[4, 5], [6, 7]], [[16, 17], [18, 19]]]),
            arr3(&[[[8, 9], [10, 11]], [[20, 21], [22, 23]]]),
        ],
    );

    let a = ArcArray::from_iter(0..28);
    let a = a.reshape((2, 7, 2));

    let it = a.axis_chunks_iter(Axis(1), 2);
    assert_equal(
        it,
        vec![
            arr3(&[[[0, 1], [2, 3]], [[14, 15], [16, 17]]]),
            arr3(&[[[4, 5], [6, 7]], [[18, 19], [20, 21]]]),
            arr3(&[[[8, 9], [10, 11]], [[22, 23], [24, 25]]]),
            arr3(&[[[12, 13]], [[26, 27]]]),
        ],
    );

    let it = a.axis_chunks_iter(Axis(1), 2).rev();
    assert_equal(
        it,
        vec![
            arr3(&[[[12, 13]], [[26, 27]]]),
            arr3(&[[[8, 9], [10, 11]], [[22, 23], [24, 25]]]),
            arr3(&[[[4, 5], [6, 7]], [[18, 19], [20, 21]]]),
            arr3(&[[[0, 1], [2, 3]], [[14, 15], [16, 17]]]),
        ],
    );

    let it = a.axis_chunks_iter(Axis(1), 7);
    assert_equal(it, vec![a.view()]);

    let it = a.axis_chunks_iter(Axis(1), 9);
    assert_equal(it, vec![a.view()]);
}

#[test]
fn axis_iter_mut_split_at() {
    let mut a = Array::from_iter(0..5);
    let mut a_clone = a.clone();
    let all: Vec<_> = a_clone.axis_iter_mut(Axis(0)).collect();
    for mid in 0..=all.len() {
        let (left, right) = a.axis_iter_mut(Axis(0)).split_at(mid);
        assert_eq!(&all[..mid], &left.collect::<Vec<_>>()[..]);
        assert_eq!(&all[mid..], &right.collect::<Vec<_>>()[..]);
    }
}

#[test]
fn axis_iter_mut_split_at_partially_consumed() {
    let mut a = Array::from_iter(0..5);
    for consumed in 1..=a.len() {
        for mid in 0..=(a.len() - consumed) {
            let mut a_clone = a.clone();
            let remaining: Vec<_> = {
                let mut iter = a_clone.axis_iter_mut(Axis(0));
                for _ in 0..consumed {
                    iter.next();
                }
                iter.collect()
            };
            let (left, right) = {
                let mut iter = a.axis_iter_mut(Axis(0));
                for _ in 0..consumed {
                    iter.next();
                }
                iter.split_at(mid)
            };
            assert_eq!(&remaining[..mid], &left.collect::<Vec<_>>()[..]);
            assert_eq!(&remaining[mid..], &right.collect::<Vec<_>>()[..]);
        }
    }
}

#[test]
fn axis_iter_mut_zip() {
    let orig = Array::from_iter(0..5);
    let mut cloned = orig.clone();
    let iter = cloned.axis_iter_mut(Axis(0));
    let mut b = Array::zeros(5);
    Zip::from(&mut b).and(iter).for_each(|b, mut a| {
        a[()] += 1;
        *b = a[()];
    });
    assert_eq!(cloned, b);
    assert_eq!(cloned, orig + 1);
}

#[test]
fn axis_iter_mut_zip_partially_consumed() {
    let mut a = Array::from_iter(0..5);
    for consumed in 1..=a.len() {
        let remaining = a.len() - consumed;
        let mut iter = a.axis_iter_mut(Axis(0));
        for _ in 0..consumed {
            iter.next();
        }
        let mut b = Array::zeros(remaining);
        Zip::from(&mut b).and(iter).for_each(|b, a| *b = a[()]);
        assert_eq!(a.slice(s![consumed..]), b);
    }
}

#[test]
fn axis_iter_mut_zip_partially_consumed_discontiguous() {
    let mut a = Array::from_iter(0..5);
    for consumed in 1..=a.len() {
        let remaining = a.len() - consumed;
        let mut iter = a.axis_iter_mut(Axis(0));
        for _ in 0..consumed {
            iter.next();
        }
        let mut b = Array::zeros(remaining * 2);
        b.slice_collapse(s![..;2]);
        Zip::from(&mut b).and(iter).for_each(|b, a| *b = a[()]);
        assert_eq!(a.slice(s![consumed..]), b);
    }
}

#[test]
#[cfg(feature = "std")]
fn axis_chunks_iter_corner_cases() {
    // examples provided by @bluss in PR #65
    // these tests highlight corner cases of the axis_chunks_iter implementation
    // and enable checking if no pointer offsetting is out of bounds. However
    // checking the absence of of out of bounds offsetting cannot (?) be
    // done automatically, so one has to launch this test in a debugger.
    let a = ArcArray::<f32, _>::linspace(0., 7., 8).reshape((8, 1));
    let it = a.axis_chunks_iter(Axis(0), 4);
    assert_equal(it, vec![a.slice(s![..4, ..]), a.slice(s![4.., ..])]);
    let a = a.slice(s![..;-1,..]);
    let it = a.axis_chunks_iter(Axis(0), 8);
    assert_equal(it, vec![a.view()]);
    let it = a.axis_chunks_iter(Axis(0), 3);
    assert_equal(
        it,
        vec![
            array![[7.], [6.], [5.]],
            array![[4.], [3.], [2.]],
            array![[1.], [0.]],
        ],
    );

    let b = ArcArray::<f32, _>::zeros((8, 2));
    let a = b.slice(s![1..;2,..]);
    let it = a.axis_chunks_iter(Axis(0), 8);
    assert_equal(it, vec![a.view()]);

    let it = a.axis_chunks_iter(Axis(0), 1);
    assert_equal(it, vec![ArcArray::zeros((1, 2)); 4]);
}

#[test]
fn axis_chunks_iter_zero_stride() {
    {
        // stride 0 case
        let b = Array::from(vec![0f32; 0]).into_shape((5, 0, 3)).unwrap();
        let shapes: Vec<_> = b
            .axis_chunks_iter(Axis(0), 2)
            .map(|v| v.raw_dim())
            .collect();
        assert_eq!(shapes, vec![Ix3(2, 0, 3), Ix3(2, 0, 3), Ix3(1, 0, 3)]);
    }

    {
        // stride 0 case reverse
        let b = Array::from(vec![0f32; 0]).into_shape((5, 0, 3)).unwrap();
        let shapes: Vec<_> = b
            .axis_chunks_iter(Axis(0), 2)
            .rev()
            .map(|v| v.raw_dim())
            .collect();
        assert_eq!(shapes, vec![Ix3(1, 0, 3), Ix3(2, 0, 3), Ix3(2, 0, 3)]);
    }

    // From issue #542, ZST element
    {
        let a = Array::from(vec![(); 3]);
        let chunks: Vec<_> = a.axis_chunks_iter(Axis(0), 2).collect();
        assert_eq!(chunks, vec![a.slice(s![0..2]), a.slice(s![2..])]);
    }
}

#[should_panic]
#[test]
fn axis_chunks_iter_zero_chunk_size() {
    let a = Array::from_iter(0..5);
    a.axis_chunks_iter(Axis(0), 0);
}

#[test]
fn axis_chunks_iter_zero_axis_len() {
    let a = Array::from_iter(0..0);
    assert!(a.axis_chunks_iter(Axis(0), 5).next().is_none());
}

#[test]
fn axis_chunks_iter_split_at() {
    let mut a = Array2::<usize>::zeros((11, 3));
    a.iter_mut().enumerate().for_each(|(i, elt)| *elt = i);
    for source in &[
        a.slice(s![..0, ..]),
        a.slice(s![..1, ..]),
        a.slice(s![..5, ..]),
        a.slice(s![..10, ..]),
        a.slice(s![..11, ..]),
        a.slice(s![.., ..0]),
    ] {
        let chunks_iter = source.axis_chunks_iter(Axis(0), 5);
        let all_chunks: Vec<_> = chunks_iter.clone().collect();
        let n_chunks = chunks_iter.len();
        assert_eq!(n_chunks, all_chunks.len());
        for index in 0..=n_chunks {
            let (left, right) = chunks_iter.clone().split_at(index);
            assert_eq!(&all_chunks[..index], &left.collect::<Vec<_>>()[..]);
            assert_eq!(&all_chunks[index..], &right.collect::<Vec<_>>()[..]);
        }
        assert_panics!({
            chunks_iter.split_at(n_chunks + 1);
        });
    }
}

#[test]
fn axis_chunks_iter_mut() {
    let a = ArcArray::from_iter(0..24);
    let mut a = a.reshape((2, 6, 2));

    let mut it = a.axis_chunks_iter_mut(Axis(1), 2);
    let mut col0 = it.next().unwrap();
    col0[[0, 0, 0]] = 42;
    assert_eq!(col0, arr3(&[[[42, 1], [2, 3]], [[12, 13], [14, 15]]]));
}

#[should_panic]
#[test]
fn axis_chunks_iter_mut_zero_chunk_size() {
    let mut a = Array::from_iter(0..5);
    a.axis_chunks_iter_mut(Axis(0), 0);
}

#[test]
fn axis_chunks_iter_mut_zero_axis_len() {
    let mut a = Array::from_iter(0..0);
    assert!(a.axis_chunks_iter_mut(Axis(0), 5).next().is_none());
}

#[test]
fn outer_iter_size_hint() {
    // Check that the size hint is correctly computed
    let a = ArcArray::from_iter(0..24).reshape((4, 3, 2));
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
    let a = ArcArray::from_iter(0..30).reshape((5, 3, 2));

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
    let a = ArcArray::from_iter(0..30).reshape((5, 3, 2));

    let it = a.outer_iter();
    it.split_at(6);
}

#[test]
fn outer_iter_mut_split_at() {
    let mut a = ArcArray::from_iter(0..30).reshape((5, 3, 2));

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

#[test]
fn iterators_are_send_sync() {
    // When the element type is Send + Sync, then the iterators and views
    // are too.
    fn _send_sync<T: Send + Sync>(_: &T) {}

    let mut a = ArcArray::from_iter(0..30).into_shape((5, 3, 2)).unwrap();

    _send_sync(&a.view());
    _send_sync(&a.view_mut());
    _send_sync(&a.iter());
    _send_sync(&a.iter_mut());
    _send_sync(&a.indexed_iter());
    _send_sync(&a.indexed_iter_mut());
    _send_sync(&a.rows());
    _send_sync(&a.rows_mut());
    _send_sync(&a.outer_iter());
    _send_sync(&a.outer_iter_mut());
    _send_sync(&a.axis_iter(Axis(1)));
    _send_sync(&a.axis_iter_mut(Axis(1)));
    _send_sync(&a.axis_chunks_iter(Axis(1), 1));
    _send_sync(&a.axis_chunks_iter_mut(Axis(1), 1));
    _send_sync(&indices(a.dim()));
    _send_sync(&a.exact_chunks((1, 1, 1)));
    _send_sync(&a.exact_chunks_mut((1, 1, 1)));
    _send_sync(&a.exact_chunks((1, 1, 1)).into_iter());
    _send_sync(&a.exact_chunks_mut((1, 1, 1)).into_iter());
}

#[test]
#[allow(clippy::unnecessary_fold)]
fn test_fold() {
    let mut a = Array2::<i32>::default((20, 20));
    a += 1;
    let mut iter = a.iter();
    iter.next();
    assert_eq!(iter.fold(0, |acc, &x| acc + x), a.sum() - 1);

    let mut a = Array0::<i32>::default(());
    a += 1;
    assert_eq!(a.iter().fold(0, |acc, &x| acc + x), 1);
}

#[test]
fn nth_back_examples() {
    let mut a: Array1<i32> = (0..256).collect();
    a.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
    assert_eq!(a.iter().nth_back(0), Some(&a[a.len() - 1]));
    assert_eq!(a.iter().nth_back(1), Some(&a[a.len() - 2]));
    assert_eq!(a.iter().nth_back(a.len() - 2), Some(&a[1]));
    assert_eq!(a.iter().nth_back(a.len() - 1), Some(&a[0]));
    assert_eq!(a.iter().nth_back(a.len()), None);
    assert_eq!(a.iter().nth_back(a.len() + 1), None);
    assert_eq!(a.iter().nth_back(a.len() + 2), None);
}

#[test]
fn nth_back_zero_n() {
    let mut a: Array1<i32> = (0..256).collect();
    a.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
    let mut iter1 = a.iter();
    let mut iter2 = a.iter();
    for _ in 0..(a.len() + 1) {
        assert_eq!(iter1.nth_back(0), iter2.next_back());
        assert_eq!(iter1.len(), iter2.len());
    }
}

#[test]
fn nth_back_nonzero_n() {
    let mut a: Array1<i32> = (0..256).collect();
    a.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
    let mut iter1 = a.iter();
    let mut iter2 = a.iter();
    for _ in 0..(a.len() / 3 + 1) {
        assert_eq!(iter1.nth_back(2), {
            iter2.next_back();
            iter2.next_back();
            iter2.next_back()
        });
        assert_eq!(iter1.len(), iter2.len());
    }
}

#[test]
fn nth_back_past_end() {
    let mut a: Array1<i32> = (0..256).collect();
    a.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
    let mut iter = a.iter();
    assert_eq!(iter.nth_back(a.len()), None);
    assert_eq!(iter.next(), None);
}

#[test]
fn nth_back_partially_consumed() {
    let mut a: Array1<i32> = (0..256).collect();
    a.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
    let mut iter = a.iter();
    iter.next();
    iter.next_back();
    assert_eq!(iter.len(), a.len() - 2);
    assert_eq!(iter.nth_back(1), Some(&a[a.len() - 3]));
    assert_eq!(iter.len(), a.len() - 4);
    assert_eq!(iter.nth_back(a.len() - 6), Some(&a[2]));
    assert_eq!(iter.len(), 1);
    assert_eq!(iter.next(), Some(&a[1]));
    assert_eq!(iter.len(), 0);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next_back(), None);
}

#[test]
fn test_rfold() {
    {
        let mut a = Array1::<i32>::default(256);
        a += 1;
        let mut iter = a.iter();
        iter.next();
        assert_eq!(iter.rfold(0, |acc, &x| acc + x), a.sum() - 1);
    }

    // Test strided arrays
    {
        let mut a = Array1::<i32>::default(256);
        a.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
        a += 1;
        let mut iter = a.iter();
        iter.next();
        assert_eq!(iter.rfold(0, |acc, &x| acc + x), a.sum() - 1);
    }

    {
        let mut a = Array1::<i32>::default(256);
        a.slice_axis_inplace(Axis(0), Slice::new(0, None, -2));
        a += 1;
        let mut iter = a.iter();
        iter.next();
        assert_eq!(iter.rfold(0, |acc, &x| acc + x), a.sum() - 1);
    }

    // Test order
    {
        let mut a = Array1::from_iter(0..20);
        a.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
        let mut iter = a.iter();
        iter.next();
        let output = iter.rfold(Vec::new(), |mut acc, elt| {
            acc.push(*elt);
            acc
        });
        assert_eq!(
            Array1::from(output),
            Array::from_iter((1..10).rev().map(|i| i * 2))
        );
    }
}

#[test]
fn test_into_iter() {
    let a = Array1::from(vec![1, 2, 3, 4]);
    let v = a.into_iter().collect::<Vec<_>>();
    assert_eq!(v, [1, 2, 3, 4]);
}

#[test]
fn test_into_iter_2d() {
    let a = Array1::from(vec![1, 2, 3, 4]).into_shape((2, 2)).unwrap();
    let v = a.into_iter().collect::<Vec<_>>();
    assert_eq!(v, [1, 2, 3, 4]);

    let a = Array1::from(vec![1, 2, 3, 4]).into_shape((2, 2)).unwrap().reversed_axes();
    let v = a.into_iter().collect::<Vec<_>>();
    assert_eq!(v, [1, 3, 2, 4]);
}

#[test]
fn test_into_iter_sliced() {
    let (m, n) = (4, 5);
    let drops = Cell::new(0);

    for i in 0..m - 1 {
        for j in 0..n - 1 {
            for i2 in i + 1 .. m {
                for j2 in j + 1 .. n {
                    for invert in 0..3 {
                        drops.set(0);
                        let i = i as isize;
                        let j = j as isize;
                        let i2 = i2 as isize;
                        let j2 = j2 as isize;
                        let mut a = Array1::from_iter(0..(m * n) as i32)
                            .mapv(|v| DropCount::new(v, &drops))
                            .into_shape((m, n)).unwrap();
                        a.slice_collapse(s![i..i2, j..j2]);
                        if invert < a.ndim() {
                            a.invert_axis(Axis(invert));
                        }

                        println!("{:?}, {:?}", i..i2, j..j2);
                        println!("{:?}", a);
                        let answer = a.iter().cloned().collect::<Vec<_>>();
                        let v = a.into_iter().collect::<Vec<_>>();
                        assert_eq!(v, answer);

                        assert_eq!(drops.get(), m * n - v.len());
                        drop(v);
                        assert_eq!(drops.get(), m * n);
                    }
                }
            }
        }
    }
}

/// Helper struct that counts its drops Asserts that it's not dropped twice. Also global number of
/// drops is counted in the cell.
///
/// Compares equal by its "represented value".
#[derive(Clone, Debug)]
struct DropCount<'a> {
    value: i32,
    my_drops: usize,
    drops: &'a Cell<usize>
}

impl PartialEq for DropCount<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<'a> DropCount<'a> {
    fn new(value: i32, drops: &'a Cell<usize>) -> Self {
        DropCount { value, my_drops: 0, drops }
    }
}

impl Drop for DropCount<'_> {
    fn drop(&mut self) {
        assert_eq!(self.my_drops, 0);
        self.my_drops += 1;
        self.drops.set(self.drops.get() + 1);
    }
}
