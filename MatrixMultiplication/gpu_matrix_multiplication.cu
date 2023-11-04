
void cudaErrorCheck(cudaError_t error, bool abort = true);

void printKernelData(dim3 blockDim, dim3 gridDim);

/* ---------------------------------------------------------------- */
__global__ void gpuKernel(const int *a, const int *b, int *c,
                          const size_t row_A, const size_t row_B,
                          const size_t col_B) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;

  if (idx < row_A && idy < col_B) {
    int value = 0;
    for (size_t k = 0; k < row_B; k++) {
      value += a[idx * row_B + k] * b[k * col_B + idy];
    }
    c[idx * col_B + idy] = value;
  }
}

/* ---------------------------------------------------------------- */
void gpuMatrixMultiplication(const int *A, const int *B, int *C,
                             const size_t row_A, const size_t row_B,
                             const size_t col_B) {
  // Creating device variables for all the three matrices
  int *a_d, *b_d, *c_d;

  // Allocating memory for device variables
  cudaErrorCheck(cudaMalloc((void **)&a_d, row_A * row_B * sizeof(int)));
  cudaErrorCheck(cudaMalloc((void **)&b_d, row_B * col_B * sizeof(int)));
  cudaErrorCheck(cudaMalloc((void **)&c_d, row_A * col_B * sizeof(int)));

  // Copying the input matrices from host to device
  cudaErrorCheck(
      cudaMemcpy(a_d, A, row_A * row_B * sizeof(int), cudaMemcpyHostToDevice));
  cudaErrorCheck(
      cudaMemcpy(b_d, B, row_B * col_B * sizeof(int), cudaMemcpyHostToDevice));

  const size_t threads_per_block = 32;
  dim3 block_dimension(threads_per_block, threads_per_block);
  dim3 grid_dimension(ceil(float(row_A) / threads_per_block),
                      ceil(float(col_B) / threads_per_block));

  printKernelData(block_dimension, grid_dimension);

  // Calling the kernel
  gpuKernel<<<grid_dimension, block_dimension>>>(a_d, b_d, c_d, row_A, row_B,
                                                 col_B);

  // Copying the output from device to host
  cudaErrorCheck(
      cudaMemcpy(C, c_d, row_A * col_B * sizeof(int), cudaMemcpyDeviceToHost));

  // Freeing the memory
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
}
