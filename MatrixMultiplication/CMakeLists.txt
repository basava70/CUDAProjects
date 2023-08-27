cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# project
project(run_matrix_multiplication CXX CUDA)

# compiling json commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# adding add_executable
add_executable(run_matrix_multiplication matrix_multiplication.cu)
