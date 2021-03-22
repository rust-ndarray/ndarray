#!/bin/sh

set -x
set -e

FEATURES=$1
CHANNEL=$2
TARGET=$3

cross build -v --features="$FEATURES" --target=$TARGET
cross test -v --no-fail-fast --features="$FEATURES" --target=$TARGET
cross test -v --no-fail-fast --target=$TARGET --manifest-path=ndarray-rand/Cargo.toml --features quickcheck
cross test -v --no-fail-fast --target=$TARGET --manifest-path=xtest-serialization/Cargo.toml --verbose
CARGO_TARGET_DIR=target/ cross test -v --no-fail-fast --target=$TARGET --manifest-path=xtest-numeric/Cargo.toml
