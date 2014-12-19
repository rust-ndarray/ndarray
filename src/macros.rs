#![macro_escape]

#[macro_export]
/// Create a 2D matrix
///
/// ## Example
///
/// ```ignore
/// let e = matrix![1., 0.; 0., 1.0_f32];
/// let m = matrix![1, 2, 3, 4; 5, 6, 7, 8i];
/// ```
///
/// **Panics** if row lengths don't match.
pub macro_rules! matrix(
    // This is a hack -- so we don't have to export another macro
    // for argument counting.
    // Count the macro arguments and evaluate to an expression
    // (a sum `1 + 1 + ... + 1 + 0`).
    (__count ) => { 0 };
    (__count $_i:tt $(, $rest:tt)*) => { 1 + matrix!(__count $($rest),*) };
    // matrix macro follows
    ($($($elt:expr),*);*) => {
        {
            // Expand the matrix row-by-row and check that the
            // dimensions match. The compiler will constant fold it all.
            let mut _first = true;
            let mut rows: ::ndarray::Ix = 0;
            let mut cols: ::ndarray::Ix = 0;
            $(
                {
                    let this_count = matrix!(__count $($elt),*);
                    if !_first && this_count != cols {
                        panic!("Row length mismatch in matrix![]")
                    }
                    _first = false;
                    cols = this_count;
                }
                rows += 1;
            )*
            unsafe {
                ::ndarray::Array::from_vec_dim(
                    (rows, cols),
                    vec![
                    $(
                        $($elt,)*
                    )*
                    ]
                )
            }
        }
    };
);


