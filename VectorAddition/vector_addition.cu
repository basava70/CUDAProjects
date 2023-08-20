/**
 * Cuda program to calculate the vector addition
 * in parallel using Cuda. Written on August 19.
 *
 **/

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
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

// gpu kernel
template <typename T> __global__ void vecAddKernel(T *a, T *b, T *c, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

/**
 * This function does all the necessary things for the gpu vector addition.
 * Creates the three device vectors a_d, b_d and c_d, allocates memory
 * and sends them to the gpu kernel with necesary threads and blocks.
 * Frees the memory from gpu as well.
 **/
template <typename T> size_t vecAdd(const T *a, const T *b, T *c, size_t n) {

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
  size_t blockDim = ceil(float(n) / threads_per_block);

  // assigning kernel
  vecAddKernel<<<blockDim, threads_per_block>>>(a_d, b_d, c_d, n);

  // copy the result device variable c_d back to the host variable c_h
  cudaErrorCheck(cudaMemcpy(c, c_d, n * sizeof(T), cudaMemcpyDeviceToHost));

  // free cuda memory
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  return threads_per_block;
}

/**
 * The cpu kernel for addition to compare computation time
 **/
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

  for (int i = 0; i < n; i++) {
    a_h[i] = i;
    b_h[i] = 2 * i;
  }

  // measuring time for the gpu version
  auto begin_gpu = std::chrono::high_resolution_clock::now();
  size_t threads_per_block = vecAdd(a_h, b_h, c_h, n);
  auto end_gpu = std::chrono::high_resolution_clock::now();

  // checking for correctness
  double tolerance = 1e-14;
  for (size_t iter = 0; iter < n; iter++) {
    if (fabs(c_h[iter] - 3.0 * iter) > tolerance) {
      std::cout << "Error c_h[" << iter << "] = " << c_h[iter] << "instead of "
                << 3 * iter << std::endl;
      exit(1);
    }
  }

  std::cout << " ------------------------ " << std::endl;
  std::cout << " ------- Success -------- " << std::endl;
  std::cout << " Threads used : " << threads_per_block << std::endl;
  std::cout << " Blocks used : " << ceil(float(n) / threads_per_block)
            << std::endl;
  std::cout << " ------------------------ " << std::endl;

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

  // free cpu memory
  free(a_h);
  free(b_h);
  free(c_h);

  return 0;
}
