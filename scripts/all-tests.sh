#!/bin/sh

set -x
set -e

FEATURES=$1
CHANNEL=$2

if [ "$CHANNEL" = "1.51.0" ]; then
    cargo update --package rayon --precise 1.5.3
    cargo update --package rayon-core --precise 1.9.3
    cargo update --package serde --precise 1.0.156
    cargo update --package thiserror --precise 1.0.39
    cargo update --package once_cell --precise 1.14.0
    cargo update --package openblas-src --precise 0.10.5
    cargo update --package openblas-build --precise 0.10.5
fi

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
cargo test --manifest-path=xtest-numeric/Cargo.toml --verbose
cargo test --manifest-path=xtest-numeric/Cargo.toml --verbose --features test_blas
([ "$CHANNEL" != "nightly" ] || cargo bench --no-run --verbose --features "$FEATURES")
