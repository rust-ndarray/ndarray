//! Mock interfaces to BLAS

use core::cell::RefCell;
use core::ffi::{c_double, c_float, c_int};
use std::thread_local;

use cblas_sys::{c_double_complex, c_float_complex, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

thread_local! {
    /// This counter is incremented every time a gemm function is called
    pub static CALL_COUNT: RefCell<usize> = RefCell::new(0);
}

#[rustfmt::skip]
#[no_mangle]
#[allow(unused)]
pub unsafe extern "C" fn cblas_sgemm(
    layout: CBLAS_LAYOUT,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: c_float,
    a: *const c_float,
    lda: c_int,
    b: *const c_float,
    ldb: c_int,
    beta: c_float,
    c: *mut c_float,
    ldc: c_int
) {
    CALL_COUNT.with(|ctx| *ctx.borrow_mut() += 1);
}

#[rustfmt::skip]
#[no_mangle]
#[allow(unused)]
pub unsafe extern "C" fn cblas_dgemm(
    layout: CBLAS_LAYOUT,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: c_double,
    a: *const c_double,
    lda: c_int,
    b: *const c_double,
    ldb: c_int,
    beta: c_double,
    c: *mut c_double,
    ldc: c_int
) {
    CALL_COUNT.with(|ctx| *ctx.borrow_mut() += 1);
}

#[rustfmt::skip]
#[no_mangle]
#[allow(unused)]
pub unsafe extern "C" fn cblas_cgemm(
    layout: CBLAS_LAYOUT,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_float_complex,
    a: *const c_float_complex,
    lda: c_int,
    b: *const c_float_complex,
    ldb: c_int,
    beta: *const c_float_complex,
    c: *mut c_float_complex,
    ldc: c_int
) {
    CALL_COUNT.with(|ctx| *ctx.borrow_mut() += 1);
}

#[rustfmt::skip]
#[no_mangle]
#[allow(unused)]
pub unsafe extern "C" fn cblas_zgemm(
    layout: CBLAS_LAYOUT,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_double_complex,
    a: *const c_double_complex,
    lda: c_int,
    b: *const c_double_complex,
    ldb: c_int,
    beta: *const c_double_complex,
    c: *mut c_double_complex,
    ldc: c_int
) {
    CALL_COUNT.with(|ctx| *ctx.borrow_mut() += 1);
}
