#!/bin/sh

set -x
set -e

FEATURES=$1
CHANNEL=$2

QC_FEAT=--features=ndarray-rand/quickcheck

# build check with no features
cargo build -v --no-default-features

# ndarray with no features
cargo test -p ndarray -v --no-default-features
# ndarray with no_std-compatible features
cargo test -p ndarray -v --no-default-features --features approx
# all with features
cargo test -v --features "$FEATURES" $QC_FEAT
# all with features and release (ignore test crates which is already optimized)
cargo test -v -p ndarray -p ndarray-rand --release --features "$FEATURES" $QC_FEAT --lib --tests

# BLAS tests
cargo test -p ndarray --lib -v --features blas
cargo test -p blas-mock-tests -v
if [ "$CHANNEL" == "1.64.0" ]; then
    cargo test -p blas-tests -v --features blas-tests/blis-system
else
    cargo test -p blas-tests -v --features blas-tests/openblas-system
fi
if [ "$CHANNEL" == "1.64.0" ]; then
    cargo test -p numeric-tests -v --features numeric-tests/test_blis
else
    cargo test -p numeric-tests -v --features numeric-tests/test_blas
fi

# Examples
cargo test --examples

# Benchmarks
([ "$CHANNEL" != "nightly" ] || cargo bench --no-run --verbose --features "$FEATURES")
