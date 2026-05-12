cmake_minimum_required(VERSION 3.14)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_executable(MeshabilityEvaluator
        ./MeshabilityEvaluator.cc
        ./main.cc)

target_link_libraries(MeshabilityEvaluator
        AlgoHex
        CLI11::CLI11)

