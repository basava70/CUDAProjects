project(convert_color_to_gray)
cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)

# find libraries
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# include headers #
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Compiler flags and definitions
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
                    -gencode=arch=compute_86,code=compute_86)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)

# specify targets
cuda_add_executable(run_convert_color_to_gray convert_color_to_gray.cu)
target_link_libraries(run_convert_color_to_gray ${OpenCV_LIBS})
