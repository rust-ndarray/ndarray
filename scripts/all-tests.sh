#!/bin/sh

set -x
set -e

FEATURES=$1
CHANNEL=$2

cargo build --verbose --no-default-features
# Testing both dev and release profiles helps find bugs, especially in low level code
cargo test --verbose --no-default-features
cargo test --release --verbose --no-default-features
cargo build --verbose --features "$FEATURES"
cargo test --verbose --features "$FEATURES"
cargo test --manifest-path=ndarray-rand/Cargo.toml --no-default-features --verbose
cargo test --manifest-path=ndarray-rand/Cargo.toml --features quickcheck --verbose
cargo test --manifest-path=xtest-serialization/Cargo.toml --verbose
cargo test --manifest-path=xtest-blas/Cargo.toml --verbose --features openblas-system
cargo test --examples
CARGO_TARGET_DIR=target/ cargo test --manifest-path=xtest-numeric/Cargo.toml --verbose
CARGO_TARGET_DIR=target/ cargo test --manifest-path=xtest-numeric/Cargo.toml --verbose --features test_blas
([ "$CHANNEL" != "nightly" ] || cargo bench --no-run --verbose --features "$FEATURES")
