#!/bin/bash

# Echo all commands before executing them
set -o xtrace
# Forbid any unset variables
set -o nounset
# Exit on any error
set -o errexit


run_kcov() {
    sh <(cargo kcov --print-install-kcov-sh)
    KCOV_BIN="./$(ls | grep kcov)/build/src/kcov"

    # Run kcov on all the test suites
    cargo kcov --all --no-default-features --output kcov-no-default-features
    cargo kcov --all --features "$FEATURES" --output kcov-features

    $KCOV_BIN --merge kcov \
         kcov-no-default-features \
         kcov-features
}

coverage_codecov() {
    if [[ "$TRAVIS_RUST_VERSION" != "stable" ]]; then
        return
    fi

    run_kcov

    bash <(curl -s https://codecov.io/bash) -s kcov
    echo "Uploaded code coverage to codecov.io"
}

main() {
    coverage_codecov
}

main
