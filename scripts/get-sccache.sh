#!/bin/sh

set -x
set -e

# WARNING: installing and compiling sccache using cargo is slow
# and compiling on Rust 1.31 will fail
# so we will download binaries

SC_URL="https://github.com/mozilla/sccache/releases/download"
SC_VER=0.2.9
SC_ARCH=x86_64
SC_PLAT=unknown-linux-musl # or apple-darwin, pc-windows-msvc

wget "$SC_URL/$SC_VER/sccache-$SC_VER-$SC_ARCH-$SC_PLAT.tar.gz" -O - | tar xvz
mv "sccache-$SC_VER-$SC_ARCH-$SC_PLAT/sccache" $TRAVIS_HOME/bin/

exit 0
