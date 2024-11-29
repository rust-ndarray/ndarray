#!/bin/sh

set -x
set -e

FEATURES=$1
CHANNEL=$2

# BLAS tests
cargo test -p blas-tests -v --features blas-tests/openblas-system
cargo test -p numeric-tests -v --features numeric-tests/test_blas
