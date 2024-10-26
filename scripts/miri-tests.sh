#!/bin/sh

set -x
set -e

# We rely on layout-dependent casts, which should be covered with #[repr(transparent)]
# This should catch if we missed that
RUSTFLAGS="-Zrandomize-layout"

# Miri reports a stacked borrow violation deep within rayon, in a crate called crossbeam-epoch
# The crate has a PR to fix this: https://github.com/crossbeam-rs/crossbeam/pull/871
# but using Miri's tree borrow mode may resolve it for now.
# Disabled until we can figure out a different rayon issue.
# MIRIFLAGS="-Zmiri-tree-borrows"

# General tests
cargo miri test -v -p ndarray -p ndarray-rand --lib

# Examples
cargo miri test --examples
