cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(run_device_properties)

# find libraries #
find_package(CUDA REQUIRED)

# set nvcc compiler
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.2/bin/nvcc)

# set json on for neovim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# specify targets
cuda_add_executable(run_device_properties device_properties.cu)
