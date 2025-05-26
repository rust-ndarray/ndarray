/// Indexing macro for Dim<[usize; N]> this
/// gets the index at `$i` in the underlying array
macro_rules! get {
    ($dim:expr_2021, $i:expr_2021) => {
        (*$dim.ix())[$i]
    };
}
macro_rules! getm {
    ($dim:expr_2021, $i:expr_2021) => {
        (*$dim.ixm())[$i]
    };
}
