#!/bin/sh

set -x
set -e

FEATURES=$1
CHANNEL=$2
TARGET=$3

cross build -v --features="$FEATURES" --target=$TARGET
cross test -v --no-fail-fast --features="$FEATURES" --target=$TARGET
cross test -v --no-fail-fast --target=$TARGET -p ndarray-rand --features ndarray-rand/quickcheck
cross test -v --no-fail-fast --target=$TARGET -p serialization-tests --verbose
cross test -v --no-fail-fast --target=$TARGET -p numeric-tests --release
