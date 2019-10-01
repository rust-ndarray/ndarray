use ndarray::prelude::*;
use ndarray::Zip;

use std::cell::Cell;
#[cfg(debug_assertions)]
use std::mem;

#[test]
fn raw_view_cast_cell() {
    // Test .cast() by creating an ArrayView<Cell<f32>>

    let mut a = Array::from_shape_fn((10, 5), |(i, j)| (i * j) as f32);
    let answer = &a + 1.;

    {
        let raw_cell_view = a.raw_view_mut().cast::<Cell<f32>>();
        let cell_view = unsafe { raw_cell_view.deref_into_view() };

        Zip::from(cell_view).apply(|elt| elt.set(elt.get() + 1.));
    }
    assert_eq!(a, answer);
}

#[test]
fn raw_view_cast_reinterpret() {
    // Test .cast() by reinterpreting u16 as [u8; 2]
    let a = Array::from_shape_fn((5, 5).f(), |(i, j)| (i as u16) << 8 | j as u16);
    let answer = a.mapv(u16::to_ne_bytes);

    let raw_view = a.raw_view().cast::<[u8; 2]>();
    let view = unsafe { raw_view.deref_into_view() };
    assert_eq!(view, answer);
}

#[test]
fn raw_view_cast_zst() {
    struct Zst;

    let a = Array::<(), _>::default((250, 250));
    let b: RawArrayView<Zst, _> = a.raw_view().cast::<Zst>();
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.as_ptr() as *const u8, b.as_ptr() as *const u8);
}

#[test]
#[should_panic]
fn raw_view_invalid_size_cast() {
    let data = [0i32; 16];
    ArrayView::from(&data[..]).raw_view().cast::<i64>();
}

#[test]
#[should_panic]
fn raw_view_mut_invalid_size_cast() {
    let mut data = [0i32; 16];
    ArrayViewMut::from(&mut data[..])
        .raw_view_mut()
        .cast::<i64>();
}

#[test]
#[cfg(debug_assertions)]
#[should_panic = "alignment mismatch"]
fn raw_view_invalid_align_cast() {
    #[derive(Copy, Clone, Debug)]
    #[repr(transparent)]
    struct A([u8; 16]);
    #[derive(Copy, Clone, Debug)]
    #[repr(transparent)]
    struct B([f64; 2]);

    unsafe {
        const LEN: usize = 16;
        let mut buffer = [0u8; mem::size_of::<A>() * (LEN + 1)];
        // Take out a slice of buffer as &[A] which is misaligned for B
        let mut ptr = buffer.as_mut_ptr();
        if ptr as usize % mem::align_of::<B>() == 0 {
            ptr = ptr.add(1);
        }

        let view = RawArrayViewMut::from_shape_ptr(LEN, ptr as *mut A);

        // misaligned cast - test debug assertion
        view.cast::<B>();
    }
}
