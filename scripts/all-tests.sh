#!/bin/sh

set -x
set -e

FEATURES=$1
CHANNEL=$2

([ "$CHANNEL" != "beta" ] || (rustup component add rustfmt && cargo fmt --all -- --check))
export RUSTFLAGS="-D warnings"
cargo build --verbose --no-default-features
# Testing both dev and release profiles helps find bugs, especially in low level code
cargo test --verbose --no-default-features
cargo test --release --verbose --no-default-features
cargo build --verbose --features "$FEATURES"
cargo test --verbose --features "$FEATURES"
cargo test --manifest-path=serialization-tests/Cargo.toml --verbose
cargo test --manifest-path=blas-tests/Cargo.toml --verbose
CARGO_TARGET_DIR=target/ cargo test --manifest-path=numeric-tests/Cargo.toml --verbose
([ "$CHANNEL" != "beta" ] || (rustup component add clippy && cargo clippy))
([ "$CHANNEL" != "nightly" ] || cargo bench --no-run --verbose --features "$FEATURES")
