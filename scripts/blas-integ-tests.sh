#!/bin/sh

set -x
set -e

# BLAS tests
cargo test -p blas-tests -v --features blas-tests/openblas-system
cargo test -p numeric-tests -v --features numeric-tests/test_blas
