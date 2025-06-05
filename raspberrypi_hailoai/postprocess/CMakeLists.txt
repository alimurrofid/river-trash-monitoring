cmake_minimum_required(VERSION 3.10)
project(river_trash_post)

add_library(yolo_hailortpp_post SHARED river_trash_post.cpp)
target_compile_features(yolo_hailortpp_post PRIVATE cxx_std_14)
