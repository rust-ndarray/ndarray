#[cfg(feature = "approx")]
use std::convert::TryFrom;

#[cfg(feature = "approx")]
use approx::assert_abs_diff_eq;
#[cfg(feature = "approx")]
use ndarray::prelude::*;

#[cfg(feature = "approx")]
fn main() {
    // Converting an array from one datatype to another is implemented with the
    // `ArrayBase::mapv()` function. We pass a closure that is applied to each
    // element independently. This allows for more control and flexiblity in
    // converting types.
    //
    // Below, we illustrate four different approaches for the actual conversion
    // in the closure.
    // - `From` ensures lossless conversions known at compile time and is the
    //   best default choice.
    // - `TryFrom` either converts data losslessly or panics, ensuring that the
    //   rest of the program does not continue with unexpected data.
    // - `as` never panics and may silently convert in a lossy way, depending
    //   on the source and target datatypes. More details can be found in the
    //   reference: https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast
    // - Using custom logic in the closure, e.g. to clip values or for NaN
    //   handling in floats.
    //
    // For a brush-up on casting between numeric types in Rust, refer to:
    // https://doc.rust-lang.org/rust-by-example/types/cast.html

    // Infallible, lossless conversion with `From`
    // The trait `std::convert::From` is only implemented for conversions that
    // can be guaranteed to be lossless at compile time. This is the safest
    // approach.
    let a_u8: Array<u8, _> = array![[1, 2, 3], [4, 5, 6]];
    let a_f32 = a_u8.mapv(|element| f32::from(element));
    assert_abs_diff_eq!(a_f32, array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    // Fallible, lossless conversion with `TryFrom`
    // `i8` numbers can be negative, in such a case, there is no perfect
    // conversion to `u8` defined. In this example, all numbers are positive and
    // in bounds and can be converted at runtime. But for unknown runtime input,
    // this would panic with the message provided in `.expect()`. Note that you
    // can also use `.unwrap()` to be more concise.
    let a_i8: Array<i8, _> = array![120, 8, 0];
    let a_u8 = a_i8.mapv(|element| u8::try_from(element).expect("Could not convert i8 to u8"));
    assert_eq!(a_u8, array![120u8, 8u8, 0u8]);

    // Unsigned to signed integer conversion with `as`
    // A real-life example of this would be coordinates on a grid.
    // A `usize` value can be larger than what fits into a `isize`, therefore,
    // it would be safer to use `TryFrom`. Nevertheless, `as` can be used for
    // either simplicity or performance.
    // The example includes `usize::MAX` to illustrate potentially undesired
    // behavior. It will be interpreted as -1 (noop-casting + 2-complement), see
    // https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast
    let a_usize: Array<usize, _> = array![1, 2, 3, usize::MAX];
    let a_isize = a_usize.mapv(|element| element as isize);
    assert_eq!(a_isize, array![1_isize, 2_isize, 3_isize, -1_isize]);

    // Simple upcasting with `as`
    // Every `u8` fits perfectly into a `u32`, therefore this is a lossless
    // conversion.
    // Note that it is up to the programmer to ensure the validity of the
    // conversion over the lifetime of a program. With type inference, subtle
    // bugs can creep in since conversions with `as` will always compile, so a
    // programmer might not notice that a prior lossless conversion became a
    // lossy conversion. With `From`, this would be noticed at compile-time and
    // with `TryFrom`, it would also be either handled or make the program
    // panic.
    let a_u8: Array<u8, _> = array![[1, 2, 3], [4, 5, 6]];
    let a_u32 = a_u8.mapv(|element| element as u32);
    assert_eq!(a_u32, array![[1u32, 2u32, 3u32], [4u32, 5u32, 6u32]]);

    // Saturating cast with `as`
    // The `as` keyword performs a *saturating cast* When casting floats to
    // ints. This means that numbers which do not fit into the target datatype
    // will silently be clipped to the maximum/minimum numbers. Since this is
    // not obvious, we discourage the intentional use of casting with `as` with
    // silent saturation and recommend a custom logic instead which makes the
    // intent clear.
    let a_f32: Array<f32, _> = array![
        256.0,         // saturated to 255
        255.7,         // saturated to 255
        255.1,         // saturated to 255
        254.7,         // rounded down to 254 by cutting the decimal part
        254.1,         // rounded down to 254 by cutting the decimal part
        -1.0,          // saturated to 0 on the lower end
        f32::INFINITY, // saturated to 255
        f32::NAN,      // converted to zero
    ];
    let a_u8 = a_f32.mapv(|element| element as u8);
    assert_eq!(a_u8, array![255, 255, 255, 254, 254, 0, 255, 0]);

    // Custom mapping logic
    // Given that we pass a closure for the conversion, we can also define
    // custom logic to e.g. replace NaN values and clip others. This also
    // makes the intent clear.
    let a_f32: Array<f32, _> = array![
        270.0,         // clipped to 200
        -1.2,          // clipped to 0
        4.7,           // rounded up to 5 instead of just stripping decimals
        f32::INFINITY, // clipped to 200
        f32::NAN,      // replaced with upper bound 200
    ];
    let a_u8_custom = a_f32.mapv(|element| {
        if element == f32::INFINITY || element.is_nan() {
            return 200;
        }
        if let Some(std::cmp::Ordering::Less) = element.partial_cmp(&0.0) {
            return 0;
        }
        200.min(element.round() as u8)
    });
    assert_eq!(a_u8_custom, array![200, 0, 5, 200, 200]);
}

#[cfg(not(feature = "approx"))]
fn main() {}
