
///
/// This build script emits the openblas linking directive if requested
///

fn main() {
    if cfg!(feature = "blas-openblas-sys") {
        println!("cargo:rustc-link-lib={}=openblas", "dylib");
    }
}
