use ndarray::prelude::*;

use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn assign() {
    let mut a = arr2(&[[1., 2.], [3., 4.]]);
    let b = arr2(&[[1., 3.], [2., 4.]]);
    a.assign(&b);
    assert_eq!(a, b);

    /* Test broadcasting */
    a.assign(&ArcArray::zeros(1));
    assert_eq!(a, ArcArray::zeros((2, 2)));

    /* Test other type */
    a.assign(&Array::from_elem((2, 2), 3.));
    assert_eq!(a, ArcArray::from_elem((2, 2), 3.));

    /* Test mut view */
    let mut a = arr2(&[[1, 2], [3, 4]]);
    {
        let mut v = a.view_mut();
        v.slice_collapse(s![..1, ..]);
        v.fill(0);
    }
    assert_eq!(a, arr2(&[[0, 0], [3, 4]]));
}


#[test]
fn assign_to() {
    let mut a = arr2(&[[1., 2.], [3., 4.]]);
    let b = arr2(&[[0., 3.], [2., 0.]]);
    b.assign_to(&mut a);
    assert_eq!(a, b);
}

#[test]
fn move_into_copy() {
    let a = arr2(&[[1., 2.], [3., 4.]]);
    let acopy = a.clone();
    let mut b = Array::uninit(a.dim());
    a.move_into_uninit(b.view_mut());
    let b = unsafe { b.assume_init() };
    assert_eq!(acopy, b);

    let a = arr2(&[[1., 2.], [3., 4.]]).reversed_axes();
    let acopy = a.clone();
    let mut b = Array::uninit(a.dim());
    a.move_into_uninit(b.view_mut());
    let b = unsafe { b.assume_init() };
    assert_eq!(acopy, b);
}

#[test]
fn move_into_owned() {
    // Test various memory layouts and holes while moving String elements.
    for &use_f_order in &[false, true] {
        for &invert_axis in &[0b00, 0b01, 0b10, 0b11] { // bitmask for axis to invert
            for &slice in &[false, true] {
                let mut a = Array::from_shape_fn((5, 4).set_f(use_f_order),
                                                 |idx| format!("{:?}", idx));
                if slice {
                    a.slice_collapse(s![1..-1, ..;2]);
                }

                if invert_axis & 0b01 != 0 {
                    a.invert_axis(Axis(0));
                }
                if invert_axis & 0b10 != 0 {
                    a.invert_axis(Axis(1));
                }

                let acopy = a.clone();
                let mut b = Array::uninit(a.dim());
                a.move_into_uninit(b.view_mut());
                let b = unsafe { b.assume_init() };

                assert_eq!(acopy, b);
            }
        }
    }
}

#[test]
fn move_into_slicing() {
    // Count correct number of drops when using move_into_uninit and discontiguous arrays (with holes).
    for &use_f_order in &[false, true] {
        for &invert_axis in &[0b00, 0b01, 0b10, 0b11] { // bitmask for axis to invert
            let counter = DropCounter::default();
            {
                let (m, n) = (5, 4);

                let mut a = Array::from_shape_fn((m, n).set_f(use_f_order), |_idx| counter.element());
                a.slice_collapse(s![1..-1, ..;2]);
                if invert_axis & 0b01 != 0 {
                    a.invert_axis(Axis(0));
                }
                if invert_axis & 0b10 != 0 {
                    a.invert_axis(Axis(1));
                }

                let mut b = Array::uninit(a.dim());
                a.move_into_uninit(b.view_mut());
                let b = unsafe { b.assume_init() };

                let total = m * n;
                let dropped_1 = total - (m - 2) * (n - 2);
                assert_eq!(counter.created(), total);
                assert_eq!(counter.dropped(), dropped_1);
                drop(b);
            }
            counter.assert_drop_count();
        }
    }
}

#[test]
fn move_into_diag() {
    // Count correct number of drops when using move_into_uninit and discontiguous arrays (with holes).
    for &use_f_order in &[false, true] {
        let counter = DropCounter::default();
        {
            let (m, n) = (5, 4);

            let a = Array::from_shape_fn((m, n).set_f(use_f_order), |_idx| counter.element());
            let a = a.into_diag();

            let mut b = Array::uninit(a.dim());
            a.move_into_uninit(b.view_mut());
            let b = unsafe { b.assume_init() };

            let total = m * n;
            let dropped_1 = total - Ord::min(m, n);
            assert_eq!(counter.created(), total);
            assert_eq!(counter.dropped(), dropped_1);
            drop(b);
        }
        counter.assert_drop_count();
    }
}

#[test]
fn move_into_0dim() {
    // Count correct number of drops when using move_into_uninit and discontiguous arrays (with holes).
    for &use_f_order in &[false, true] {
        let counter = DropCounter::default();
        {
            let (m, n) = (5, 4);

            // slice into a 0-dim array
            let a = Array::from_shape_fn((m, n).set_f(use_f_order), |_idx| counter.element());
            let a = a.slice_move(s![2, 2]);

            assert_eq!(a.ndim(), 0);
            let mut b = Array::uninit(a.dim());
            a.move_into_uninit(b.view_mut());
            let b = unsafe { b.assume_init() };

            let total = m * n;
            let dropped_1 = total - 1;
            assert_eq!(counter.created(), total);
            assert_eq!(counter.dropped(), dropped_1);
            drop(b);
        }
        counter.assert_drop_count();
    }
}

#[test]
fn move_into_empty() {
    // Count correct number of drops when using move_into_uninit and discontiguous arrays (with holes).
    for &use_f_order in &[false, true] {
        let counter = DropCounter::default();
        {
            let (m, n) = (5, 4);

            // slice into an empty array;
            let a = Array::from_shape_fn((m, n).set_f(use_f_order), |_idx| counter.element());
            let a = a.slice_move(s![..0, 1..1]);
            assert!(a.is_empty());
            let mut b = Array::uninit(a.dim());
            a.move_into_uninit(b.view_mut());
            let b = unsafe { b.assume_init() };

            let total = m * n;
            let dropped_1 = total;
            assert_eq!(counter.created(), total);
            assert_eq!(counter.dropped(), dropped_1);
            drop(b);
        }
        counter.assert_drop_count();
    }
}

#[test]
fn move_into() {
    // Test various memory layouts and holes while moving String elements with move_into
    for &use_f_order in &[false, true] {
        for &invert_axis in &[0b00, 0b01, 0b10, 0b11] { // bitmask for axis to invert
            for &slice in &[false, true] {
                let mut a = Array::from_shape_fn((5, 4).set_f(use_f_order),
                                                 |idx| format!("{:?}", idx));
                if slice {
                    a.slice_collapse(s![1..-1, ..;2]);
                }

                if invert_axis & 0b01 != 0 {
                    a.invert_axis(Axis(0));
                }
                if invert_axis & 0b10 != 0 {
                    a.invert_axis(Axis(1));
                }

                let acopy = a.clone();
                let mut b = Array::default(a.dim().set_f(!use_f_order ^ !slice));
                a.move_into(&mut b);

                assert_eq!(acopy, b);
            }
        }
    }
}


/// This counter can create elements, and then count and verify
/// the number of which have actually been dropped again.
#[derive(Default)]
struct DropCounter {
    created: AtomicUsize,
    dropped: AtomicUsize,
}

struct Element<'a>(&'a AtomicUsize);

impl DropCounter {
    fn created(&self) -> usize {
        self.created.load(Ordering::Relaxed)
    }

    fn dropped(&self) -> usize {
        self.dropped.load(Ordering::Relaxed)
    }

    fn element(&self) -> Element<'_> {
        self.created.fetch_add(1, Ordering::Relaxed);
        Element(&self.dropped)
    }

    fn assert_drop_count(&self) {
        assert_eq!(
            self.created(),
            self.dropped(),
            "Expected {} dropped elements, but found {}",
            self.created(),
            self.dropped()
        );
    }
}

impl<'a> Drop for Element<'a> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}
