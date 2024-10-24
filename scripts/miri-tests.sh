#!/bin/sh

set -x
set -e

# We rely on layout-dependent casts, which should be covered with #[repr(transparent)]
# This should catch if we missed that
RUSTFLAGS="-Zrandomize-layout"

# General tests
cargo miri test -v --no-default-features
cargo miri test -v -p ndarray -p ndarray-rand --lib --tests

# BLAS tests
cargo miri test -p ndarray --lib -v --features blas
cargo miri test -p blas-mock-tests -v
cargo miri test -p blas-tests -v --features blas-tests/openblas-system
cargo miri test -p numeric-tests -v --features numeric-tests/test_blas

# Examples
cargo miri test --examples
