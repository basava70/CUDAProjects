cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(run_blur_image CXX CUDA)

# find packages
find_package(OpenCV REQUIRED)

# include headers
include_directories(${OpenCV_INCLUDE_DIRS})

# specify targets
add_executable(run_blur_image blur_image.cu)
target_link_libraries(run_blur_image ${OpenCV_LIBS})
