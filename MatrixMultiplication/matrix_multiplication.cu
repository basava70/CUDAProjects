/**
 * In this project, we perform matrix multiplication
 * using CUDA. Created on August 27.
 *
 * -------------------------------------------------------*
 * Given two matrices A and B of size m *n and n * o respectively,
 * we get A * B = C which is of the size m * o.
 * The mandatory rule for the matrix multiplication is that the
 * number of columns of matrix A must match the number of rows of
 * matrix B. The size of matrix C is always number of rows of A *
 * number of columns of B.
 *
 * -------------------------------------------------------*
 * For example, consider the matrices
 *     -----------                  --------
 * A = | 1  2  3 |             B = | 7  10 |
 *     | 4  5  6 |                 | 8  11 |
 *     -----------                 | 9  12 |
 *                                 --------
 * We have C(0,0) = A[1, : ] * B[:,1]
 * C(0,0) = 1 * 7 + 2 * 8 + 3 * 9
 *
 * -------------------------------------------------------*
 * Since in C/C++, a two dimensional array or matrix is
 * represented as row major array, i.e., A = [ 1  2  3  4  5  6 ]
 * B = [ 7  10  8  11  9  12 ]. We use " j * cols + i "
 * operation to identify an element (i,j) of a matrix, i.e.,
 * A(i,j) = A[j* rows + i]; where "cols" is the number of columns of
 * the given matrix
 *
 **/

#include <cstddef>
#include <iostream>
#include <stdio.h>
#include <time.h>

size_t row_A = 512, row_B = 512, col_A = 512, col_B = 512;
// size_t row_A = 3, row_B = 3, col_A = 3, col_B = 3;

// Error handling
inline void cudaErrorCheck(cudaError_t error, bool abort = true) {
  if (error != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__,
           __LINE__);
    if (abort)
      exit(EXIT_FAILURE);
  }
}

void printKernelData(dim3 blockDim, dim3 gridDim) {
  std::cout << "blockDim(x, y, z) : (" << blockDim.x << ", " << blockDim.y
            << ", " << blockDim.z << ") " << std::endl;
  std::cout << "gridDim(x, y, z) : (" << gridDim.x << ", " << gridDim.y << ", "
            << gridDim.z << ") " << std::endl;
}

// Initialize the matrices A and B randomly between -50 to 50
void initialize_data(int *A, int *B, int *C_cpu, int *C_gpu) {
  // Initialize to a random seed
  srand(22);
  for (size_t i = 0; i < row_A; i++) {
    for (size_t j = 0; j < col_A; j++) {
      A[j * col_A + i] = rand() % 10;
      // A[j * col_A + i] = rand() % 101 - 50;
    }
  }

  for (size_t i = 0; i < row_B; i++) {
    for (size_t j = 0; j < col_B; j++) {
      B[j * col_B + i] = rand() % 10;
      // B[j * col_B + i] = rand() % 101 - 50;
    }
  }

  for (size_t i = 0; i < row_A; i++) {
    for (size_t j = 0; j < col_B; j++) {
      C_cpu[j * col_B + i] = 0;
      C_gpu[j * col_B + i] = 0;
    }
  }
}

__global__ void gpuKernel(const int *a, const int *b, int *c, const int row_b,
                          const size_t rows, const size_t cols) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = threadIdx.y + blockIdx.y * blockDim.y;

  if (idx < rows && idy < cols) {
    int value = 0;
    for (size_t i = 0; i < row_b; i++) {
      value += a[i * row_b + idx] * b[idy * cols + i];
    }
    c[idy * cols + idx] = value;
  }
}

void gpuMatrixMultiplication(const int *A, const int *B, int *C) {

  // Creating device variables for all the three matrices
  int *a_d, *b_d, *c_d;

  // Allocating memory for device variables
  cudaErrorCheck(cudaMalloc((void **)&a_d, row_A * col_A * sizeof(int)));
  cudaErrorCheck(cudaMalloc((void **)&b_d, row_B * col_B * sizeof(int)));
  cudaErrorCheck(cudaMalloc((void **)&c_d, row_A * col_B * sizeof(int)));

  cudaErrorCheck(
      cudaMemcpy(a_d, A, row_A * col_A * sizeof(int), cudaMemcpyHostToDevice));
  cudaErrorCheck(
      cudaMemcpy(b_d, B, row_B * col_B * sizeof(int), cudaMemcpyHostToDevice));

  const size_t threads_per_block = 16;
  dim3 block_dimension(threads_per_block, threads_per_block);
  dim3 grid_dimension(ceil(float(row_A) / threads_per_block),
                      ceil(float(col_B) / threads_per_block));

  printKernelData(block_dimension, grid_dimension);

  const size_t row_b = row_B;

  gpuKernel<<<grid_dimension, block_dimension>>>(a_d, b_d, c_d, row_b, row_A,
                                                 col_B);

  cudaErrorCheck(
      cudaMemcpy(C, c_d, row_A * col_B * sizeof(int), cudaMemcpyDeviceToHost));

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
}

void cpuMatrixMultiplication(const int *A, const int *B, int *C) {
  int value = 0;
  for (size_t i = 0; i < row_A; i++) {
    for (size_t j = 0; j < col_B; j++) {
      value = 0;
      for (size_t k = 0; k < row_B; k++) {
        value += A[k * col_A + i] * B[j * col_B + k];
      }
      C[j * col_B + i] = value;
    }
  }
}

void error_check(const int *C_h, const int *C_d) {
  const double tolerance = 1e-4;
  for (size_t i = 0; i < row_A; i++) {
    for (size_t j = 0; j < row_A; j++) {
      if (C_h[j * col_B + i] - C_d[j * col_B + i] > tolerance) {
        std::cout << "Error C_cpu(" << i << ", " << j
                  << ") = " << C_h[j * col_B + i] << " and C_gpu(" << i << ", "
                  << j << ") = " << C_d[j * col_B + i] << std::endl;
        exit(1);
      }
    }
  }
}

void print_matrix(const int *A) {
  for (size_t i = 0; i < row_A; i++) {
    for (size_t j = 0; j < row_A; j++) {
      std::cout << " " << A[j * row_A + i];
    }
    std::cout << std::endl;
  }
}

int main() {

  int A[row_A * col_A], B[row_B * col_B], C_cpu[row_A * col_B];

  int *C_gpu;
  C_gpu = (int *)malloc(row_A * col_B * sizeof(int));

  initialize_data(&A[0], &B[0], &C_cpu[0], C_gpu);

  gpuMatrixMultiplication(&A[0], &B[0], C_gpu);

  cpuMatrixMultiplication(&A[0], &B[0], &C_cpu[0]);

  /* std::cout << " A =" << std::endl;
  print_matrix(&A[0]);
  std::cout << " B =" << std::endl;
  print_matrix(&B[0]);
  std::cout << " C_cpu =" << std::endl;
  print_matrix(&C_cpu[0]);
  std::cout << " C_gpu =" << std::endl;
  print_matrix(C_gpu); */

  error_check(&C_cpu[0], &C_gpu[0]);

  free(C_gpu);

  return 0;
}
