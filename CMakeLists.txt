cmake_minimum_required(VERSION 3.18)
project(iknet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(iknet
    src/main.cpp
)

target_include_directories(iknet PUBLIC include)

add_executable(test_tensor test/test_tensor.cpp)
target_include_directories(test_tensor PUBLIC include)
target_link_libraries(test_tensor iknet)
