
///
/// This build script Checks if we can use #[deprecated]
///

extern crate rustc_version;

use rustc_version::Channel;

const DEPRECATED_CFG: &'static str = "has_deprecated";
const ASSIGN_FEATURE: &'static str = r#"feature="assign_ops""#;
const ASSIGN_CFG: &'static str = "has_assign";

fn main() {
    let version = rustc_version::version_meta();
    println!("cargo:rerun-if-changed=build.rs");
    if version.channel == Channel::Nightly {
        if let Some(ref date) = version.commit_date { 
            // parse year, month, day
            let ndate = date.splitn(3, "-")
                                      .map(str::parse)
                                      .collect::<Result<Vec<i32>, _>>().unwrap();

            // deprecated is available from nightly 2015-12-18
            if ndate >= vec![2015, 12, 18] {
                println!("cargo:rustc-cfg={}", DEPRECATED_CFG);
            }
            // assign_ops is available from nightly 2016-03-01
            if ndate >= vec![2016, 3, 1] {
                println!("cargo:rustc-cfg={}", ASSIGN_CFG);
                println!("cargo:rustc-cfg={}", ASSIGN_FEATURE);
            }
        }
    } else {
        if rustc_version::version_matches(">= 1.8") {
            println!("cargo:rustc-cfg={}", ASSIGN_FEATURE);
            println!("cargo:rustc-cfg={}", ASSIGN_CFG);
        }
    }
    if cfg!(feature = "blas-openblas-sys") {
        println!("cargo:rustc-link-lib={}=openblas", "dylib");
    }
}
