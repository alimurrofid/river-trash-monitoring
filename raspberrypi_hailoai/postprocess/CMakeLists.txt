cmake_minimum_required(VERSION 3.10)
project(river_trash_postprocess)

set(CMAKE_CXX_STANDARD 17)

# Include header path dari environment variable CPATH
include_directories($ENV{CPATH})

# Link path dari environment variable LD_LIBRARY_PATH
link_directories($ENV{LD_LIBRARY_PATH})

add_library(river_trash_post SHARED river_trash_post.cpp)

target_link_libraries(river_trash_post hailo_tappas_core)
