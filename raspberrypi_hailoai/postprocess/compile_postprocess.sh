#!/bin/bash

set -e

# pastikan environment hailo aktif
if [ -z "$CPATH" ]; then
  echo "ERROR: Please run 'source /home/rivermonitor/Public/hailo-rpi5-examples/setup_env.sh' first!"
  exit 1
fi

mkdir -p build
cd build

cmake ..
make

echo "Build finished. Shared object is at build/libriver_trash_post.so"
