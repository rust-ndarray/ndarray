#!/bin/sh

set -x
set -e

FEATURES=$1
IS_NIGHTLY=$2

cargo build --verbose --no-default-features
cargo test --verbose --no-default-features
cargo test --release --verbose --no-default-features
cargo build --verbose --features "$FEATURES"
cargo test --verbose --features "$FEATURES"
cargo test --manifest-path=serialization-tests/Cargo.toml --verbose
cargo test --manifest-path=blas-tests/Cargo.toml --verbose
CARGO_TARGET_DIR=target/ cargo test --manifest-path=numeric-tests/Cargo.toml --verbose
([ "$IS_NIGHTLY" != 1 ] || cargo clippy)
([ "$IS_NIGHTLY" != 1 ] || cargo bench --no-run --verbose --features "$FEATURES")
