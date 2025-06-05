#!/bin/bash

# Script ini untuk compile postprocess

mkdir -p build
cd build

cmake ..
make
