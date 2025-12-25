cmake_minimum_required(VERSION 3.18)
project(iknet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(iknet 
    src/tensor.cpp 
)

target_include_directories(iknet PUBLIC include)
