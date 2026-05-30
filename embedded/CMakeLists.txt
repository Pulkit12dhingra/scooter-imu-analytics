cmake_minimum_required(VERSION 3.16)
project(edge_scoot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT MSVC)
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# Find ONNX Runtime C++ headers and library
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        ${ONNXRUNTIME_ROOT}/include
        ${ONNXRUNTIME_ROOT}/include/onnxruntime
        /opt/homebrew/include/onnxruntime
        /usr/local/include/onnxruntime
        /opt/homebrew/include
        /usr/local/include
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS
        ${ONNXRUNTIME_ROOT}/lib
        /opt/homebrew/lib
        /usr/local/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime C++ library or headers not found. Install via 'brew install onnxruntime' or configure ONNXRUNTIME_ROOT.")
endif()

message(STATUS "Found ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Found ONNX Runtime library: ${ONNXRUNTIME_LIB}")

include_directories(
    include
    ${ONNXRUNTIME_INCLUDE_DIR}
)

add_library(embedded_core STATIC
    src/csv_parser.cpp
    src/feature_extractor.cpp
    src/inference_engine.cpp
)
target_link_libraries(embedded_core PUBLIC ${ONNXRUNTIME_LIB})

add_executable(edge_scoot
    src/main.cpp
)
target_link_libraries(edge_scoot PRIVATE embedded_core)

add_executable(feature_runner
    src/feature_runner.cpp
)
target_link_libraries(feature_runner PRIVATE embedded_core)

