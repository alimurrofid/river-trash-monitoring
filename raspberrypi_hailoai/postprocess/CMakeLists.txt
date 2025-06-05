cmake_minimum_required(VERSION 3.10)
project(river_trash_post)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include header dari folder post process hailo (pakai env var)
include_directories($ENV{TAPPAS_POST_PROC_DIR})

add_executable(river_trash_post river_trash_post.cpp)
