use ndarray::prelude::*;
use ndarray::Zip;

use std::cell::Cell;

#[test]
fn raw_view_cast_cell() {
    // Test .cast() by creating an ArrayView<Cell<f32>>

    let mut a = Array::from_shape_fn((10, 5), |(i, j)| (i * j) as f32);
    let answer = &a + 1.;

    {
        let raw_cell_view = a.raw_view_mut().cast::<Cell<f32>>();
        let cell_view = unsafe { raw_cell_view.deref_into_view() };

        Zip::from(cell_view).for_each(|elt| elt.set(elt.get() + 1.));
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
fn raw_view_misaligned() {
    let data: [u16; 2] = [0x0011, 0x2233];
    let ptr: *const u16 = data.as_ptr();
    unsafe {
        let misaligned_ptr = (ptr as *const u8).add(1) as *const u16;
        RawArrayView::from_shape_ptr(1, misaligned_ptr);
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic = "The pointer must be aligned."]
fn raw_view_deref_into_view_misaligned() {
    fn misaligned_deref(data: &[u16; 2]) -> ArrayView1<'_, u16> {
        let ptr: *const u16 = data.as_ptr();
        unsafe {
            let misaligned_ptr = (ptr as *const u8).add(1) as *const u16;
            let raw_view = RawArrayView::from_shape_ptr(1, misaligned_ptr);
            raw_view.deref_into_view()
        }
    }
    let data: [u16; 2] = [0x0011, 0x2233];
    misaligned_deref(&data);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic = "Unsupported"]
fn raw_view_negative_strides() {
    fn misaligned_deref(data: &[u16; 2]) -> ArrayView1<'_, u16> {
        let ptr: *const u16 = data.as_ptr();
        unsafe {
            let raw_view = RawArrayView::from_shape_ptr(1.strides((-1isize) as usize), ptr);
            raw_view.deref_into_view()
        }
    }
    let data: [u16; 2] = [0x0011, 0x2233];
    misaligned_deref(&data);
}
