#!/bin/bash

# Echo all commands before executing them
set -o xtrace
# Forbid any unset variables
set -o nounset
# Exit on any error
set -o errexit

KCOV_BIN="./target/kcov-master/build/src/kcov"

run_kcov() {
    # Run kcov on all the test suites
    cargo coverage --all --tests --benches --no-default-features
    mv target/kcov target/kcov-no-default-features
    cargo coverage --all --tests --benches --features "$FEATURES"
    mv target/kcov target/kcov-features

    $KCOV_BIN --merge target/kcov \
              target/kcov-no-default-features \
              target/kcov-features

    rm -rf \
       target/kcov-no-default-features \
       target/kcov-features
}

coverage_codecov() {
    if [[ "$TRAVIS_RUST_VERSION" != "stable" ]]; then
        return
    fi

    run_kcov

    bash <(curl -s https://codecov.io/bash) -s target/kcov
    echo "Uploaded code coverage to codecov.io"
}

main() {
    coverage_codecov
}

main
