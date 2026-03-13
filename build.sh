#!/bin/sh

export CoinUtils_DIR=/opt/coin-or
export IPOPT_HOME=/opt/coin-or
export CBC_DIR=/opt/coin-or
export CLP_DIR=/opt/coin-or

mkdir -p ./build && cd ./build && \
    cmake -G Ninja \
    -D CMAKE_BUILD_TYPE=RelWithDebInfo \
    -D BONMIN_ROOT_DIR=/opt/coin-or \
    ..

ninja

ln -sf "$PWD"/Build/bin/* /usr/local/bin
