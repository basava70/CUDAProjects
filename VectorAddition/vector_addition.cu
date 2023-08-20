/**
 * Cuda program to calculate the vector addition
 * in parallel using Cuda. Written on August 19.
 *
 **/

#include <chrono>
#include <iostream>
#include <stdio.h>
#include <string>

// error handling
inline void cudaErrorCheck(cudaError_t error, bool abort = true) {
  if (error != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__,
           __LINE__);
    if (abort)
      exit(EXIT_FAILURE);
  }
}

template <typename T> __global__ void vecAddKernel(T *a, T *b, T *c, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

template <typename T> void vecAdd(const T *a, const T *b, T *c, size_t n) {

  // creating device variables
  T *a_d, *b_d, *c_d;

  // allocating space for device variables
  cudaErrorCheck(cudaMalloc((void **)&a_d, n * sizeof(T)));
  cudaErrorCheck(cudaMalloc((void **)&b_d, n * sizeof(T)));
  cudaErrorCheck(cudaMalloc((void **)&c_d, n * sizeof(T)));

  // copy a and b from host to device variable a_d and b_d
  cudaErrorCheck(cudaMemcpy(a_d, a, n * sizeof(T), cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpy(b_d, b, n * sizeof(T), cudaMemcpyHostToDevice));

  // variables for defining kernels
  size_t threads_per_block = 256;

  // assigning kernel
  vecAddKernel<<<ceil(float(n) / threads_per_block), threads_per_block>>>(
      a_d, b_d, c_d, n);

  // copy the result device variable c_d back to the host variable c_h
  cudaErrorCheck(cudaMemcpy(c, c_d, n * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T> void vecAddCPU(const T *a, const T *b, T *c, size_t n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

int main() {

  // create host variables
  float *a_h, *b_h, *c_h;
  size_t n = 1000000; // size of the array
  size_t size = n * sizeof(float);

  // allocating space for the host variables
  a_h = (float *)malloc(size);
  b_h = (float *)malloc(size);
  c_h = (float *)malloc(size);

  // measuring time for the gpu version
  auto begin_gpu = std::chrono::high_resolution_clock::now();
  vecAdd(a_h, b_h, c_h, n);
  auto end_gpu = std::chrono::high_resolution_clock::now();

  std::cout << "Time elapsed for gpu is: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu -
                                                                    begin_gpu)
                   .count()
            << std::endl;

  // measuring time for the cpu version
  auto begin_cpu = std::chrono::high_resolution_clock::now();
  vecAddCPU(a_h, b_h, c_h, n);
  auto end_cpu = std::chrono::high_resolution_clock::now();

  std::cout << "Time elapsed for cpu is: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu -
                                                                    begin_cpu)
                   .count()
            << std::endl;

  return 0;
}
