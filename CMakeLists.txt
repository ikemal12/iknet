cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(iknet LANGUAGES CXX)
add_library(iknet INTERFACE)

target_include_directories(iknet INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(test_tensor test/core/test_tensor.cpp)
target_include_directories(test_tensor PUBLIC include)
target_link_libraries(test_tensor iknet GTest::gtest GTest::gtest_main)

enable_testing()
add_subdirectory(test)
