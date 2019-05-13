#!/bin/bash

# Echo all commands before executing them
set -o xtrace
# Forbid any unset variables
set -o nounset
# Exit on any error
set -o errexit

# Ensure there are no outstanding lints.
check_lints() {
    ## In the future, it would be good if `|| true` can be removed so that
    ## clippy warnings abort the build.
    cargo clippy --all --features "$FEATURES" || true
}

# Ensure the code is correctly formatted.
check_format() {
    cargo fmt --all -- --check
}

# Run the test suite.
check_tests() {
    cargo test --all --examples --tests --benches --no-default-features
    cargo test --all --examples --tests --benches --features "$FEATURES"
}

main() {
    check_lints
    check_format
    check_tests
}

main
