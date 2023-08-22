cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(run_convert_color_to_gray CXX CUDA)

# find packages
find_package(OpenCV REQUIRED)

# include headers
include_directories(${OpenCV_INCLUDE_DIRS})

# specify targets
add_executable(run_convert_color_to_gray convert_color_to_gray.cu)
target_link_libraries(run_convert_color_to_gray ${OpenCV_LIBS})
