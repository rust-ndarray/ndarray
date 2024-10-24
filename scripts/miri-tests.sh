#!/bin/sh

set -x
set -e

# We rely on layout-dependent casts, which should be covered with #[repr(transparent)]
# This should catch if we missed that
RUSTFLAGS="-Zrandomize-layout"

# Miri reports a stacked borrow violation deep within rayon, in a crate called crossbeam-epoch
# The crate has a PR to fix this: https://github.com/crossbeam-rs/crossbeam/pull/871
# but using Miri's tree borrow mode may resolve it for now
MIRIFLAGS="-Zmiri-tree-borrows"

# General tests
cargo miri test -v --features "$FEATURES"

# BLAS tests
cargo miri test -p ndarray --lib -v --features blas
cargo miri test -p blas-mock-tests -v
cargo miri test -p blas-tests -v --features blas-tests/openblas-system
cargo miri test -p numeric-tests -v --features numeric-tests/test_blas

# Examples
cargo miri test --examples
