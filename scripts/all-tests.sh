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
cargo test -p ndarray-rand --no-default-features --verbose
cargo test -p ndarray-rand --features ndarray-rand/quickcheck --verbose

cargo test -p serialization-tests -v
cargo test -p blas-tests -v --features blas-tests/openblas-system
cargo test -p numeric-tests -v
cargo test -p numeric-tests -v --features numeric-tests/test_blas

cargo test --examples
([ "$CHANNEL" != "nightly" ] || cargo bench --no-run --verbose --features "$FEATURES")
