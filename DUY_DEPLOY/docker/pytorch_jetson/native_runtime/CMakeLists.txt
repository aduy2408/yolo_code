cmake_minimum_required(VERSION 3.18)
project(trt_yolo_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include /usr/local/include /usr/local/cuda/include)
find_library(TENSORRT_LIBRARY nvinfer
    PATHS /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib /usr/local/cuda/lib64)
find_library(CUDART_LIBRARY cudart
    PATHS /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT CUDART_LIBRARY)
    message(FATAL_ERROR "TensorRT headers/library and CUDA runtime are required")
endif()

add_executable(trt_yolo_runtime trt_yolo_runtime.cpp)
target_include_directories(trt_yolo_runtime PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(trt_yolo_runtime PRIVATE ${TENSORRT_LIBRARY} ${CUDART_LIBRARY} ${OpenCV_LIBS})
