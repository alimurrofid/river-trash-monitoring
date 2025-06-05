#!/bin/bash

set -e
cd "$(dirname "$0")"
mkdir -p build && cd build
cmake ..
make

cp libyolo_hailortpp_post.so ../../resources/
echo "Shared object created at: ../../resources/libyolo_hailortpp_post.so"
