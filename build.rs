
///
/// This build script Checks if we can use #[deprecated]
///

extern crate rustc_version;

use rustc_version::Channel;

const DEPRECATED_CFG: &'static str = "has_deprecated";

fn main() {
    let version = rustc_version::version_meta();
    println!("cargo:rerun-if-changed=build.rs");
    if version.channel == Channel::Nightly || version.channel == Channel::Beta {
        if let Some(ref date) = version.commit_date { 
            // parse year, month, day
            let ndate = date.splitn(3, "-")
                                      .map(str::parse)
                                      .collect::<Result<Vec<i32>, _>>().unwrap();

            // deprecated is stable from nightly 04-12
            if ndate >= vec![2016, 04, 12] {
                println!("cargo:rustc-cfg={}", DEPRECATED_CFG);
            }
        }
    }
    if cfg!(feature = "blas-openblas-sys") {
        println!("cargo:rustc-link-lib={}=openblas", "dylib");
    }
}
