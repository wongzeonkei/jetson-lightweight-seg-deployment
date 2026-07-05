cmake_minimum_required(VERSION 3.10)
project(yolov8n_seg_trt_cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(TENSORRT_ROOT "/opt/tensorrt/TensorRT-8.6.1.6")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib")

set(CUDA_ROOT "/usr/local/cuda-11.8")
set(CUDA_INCLUDE_DIR "${CUDA_ROOT}/include")
set(CUDA_LIB_DIR "${CUDA_ROOT}/lib64")

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "TensorRT include dir: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib dir: ${TENSORRT_LIB_DIR}")
message(STATUS "CUDA include dir: ${CUDA_INCLUDE_DIR}")
message(STATUS "CUDA lib dir: ${CUDA_LIB_DIR}")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
)

add_executable(check_trt_opencv check_trt_opencv.cpp)

target_link_libraries(check_trt_opencv
    ${OpenCV_LIBS}
    ${TENSORRT_LIB_DIR}/libnvinfer.so
    ${CUDA_LIB_DIR}/libcudart.so
)

set_target_properties(check_trt_opencv PROPERTIES
    BUILD_RPATH "${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}"
)

add_executable(infer_yolov8n_seg_trt infer_yolov8n_seg_trt.cpp)

target_link_libraries(infer_yolov8n_seg_trt
    ${OpenCV_LIBS}
    ${TENSORRT_LIB_DIR}/libnvinfer.so
    ${CUDA_LIB_DIR}/libcudart.so
)

set_target_properties(infer_yolov8n_seg_trt PROPERTIES
    BUILD_RPATH "${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}"
)
