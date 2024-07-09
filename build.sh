#!/bin/bash
set -e

cmake -S . -B build/ -Dcudf_ROOT="$HOME/project/cudf/cpp/build"
cd build/
make -j$(nproc)
