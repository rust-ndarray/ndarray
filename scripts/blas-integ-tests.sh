#!/bin/sh

set -x
set -e

# BLAS tests
cargo nextest run -p blas-tests -v --features blas-tests/openblas-system
cargo nextest run -p numeric-tests -v --features numeric-tests/test_blas
