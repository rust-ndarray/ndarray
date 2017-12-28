
///
/// This build script emits the openblas linking directive if requested
///

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if cfg!(feature = "test-blas-openblas-sys") {
        println!("cargo:rustc-link-lib={}=openblas", "dylib");
    }
}
