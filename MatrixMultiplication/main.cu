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

/* ---------------------------------------------------------------- */
#include "utility.h"
#include <chrono>
#include <iostream>

/* running a testKernel to make sure we get correct time estimate
 * for the next kernels */
__global__ void testKernel() {}

/* ---------------------------------------------------------------- */
int main() {
  // const size_t row_A = 1024, row_B = 2048, col_A = 2048, col_B = 728;
  // const size_t row_A = 512, row_B = 256, col_A = 256, col_B = 512;
  const size_t row_A = 1024, row_B = 2048, col_A = 2048, col_B = 1024;
  // const size_t row_A = 512, row_B = 256, col_A = 256, col_B = 512;
  // const size_t row_A = 3, row_B = 3, col_A = 3, col_B = 7;
  // const size_t row_A = 2048, row_B = 2048, col_A = 2048, col_B = 2048;

  int *A, *B, *C_cpu, *C_gpu;
  A = reinterpret_cast<int *>(malloc(row_A * col_A * sizeof(int)));
  B = reinterpret_cast<int *>(malloc(row_B * col_B * sizeof(int)));
  C_cpu = reinterpret_cast<int *>(malloc(row_A * col_B * sizeof(int)));
  C_gpu = reinterpret_cast<int *>(malloc(row_A * col_B * sizeof(int)));

  if (row_B != col_A) {
    std::cout << " row_b != col_A " << std::endl;
    exit(1);
  }

  std::cout << " ----------------------------- " << std::endl;
  std::cout << " --- Matrix Multiplication --- " << std::endl;
  std::cout << " ----------------------------- " << std::endl;
  std::cout << " A : " << row_A << " X " << col_A << std::endl;
  std::cout << " B : " << row_B << " X " << col_B << std::endl;
  std::cout << " C : " << row_A << " X " << col_B << std::endl;

  // First kernel always takes more time
  // so calling a dummy kernel before
  testKernel<<<1, 1>>>();

  initialize_data(A, B, C_cpu, C_gpu, row_A, row_B, col_B);

  // cpu matrix multiplication
  auto begin_cpu = std::chrono::high_resolution_clock::now();
  cpuMatrixMultiplication(A, B, C_cpu, row_A, row_B, col_B);
  auto end_cpu = std::chrono::high_resolution_clock::now();

  std::cout << "Time elapsed for cpu is: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu -
                                                                     begin_cpu)
                       .count() /
                   1e6
            << " seconds " << std::endl;

  // Basic gpu matrix multiplication
  auto begin_gpu = std::chrono::high_resolution_clock::now();
  gpuMatrixMultiplication(A, B, C_gpu, row_A, row_B, col_B);
  auto end_gpu = std::chrono::high_resolution_clock::now();

  std::cout << "Time elapsed for basic gpu is: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_gpu -
                                                                     begin_gpu)
                       .count() /
                   1e6
            << " seconds " << std::endl;

  error_check(C_cpu, C_gpu, row_A, col_B);

  // Tiled gpu matrix multiplication
  auto begin_tiled_gpu = std::chrono::high_resolution_clock::now();
  gpuTiledMatrixMultiplication(A, B, C_gpu, row_A, row_B, col_B);
  auto end_tiled_gpu = std::chrono::high_resolution_clock::now();

  std::cout << "Time elapsed for tiled gpu is: "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   end_tiled_gpu - begin_tiled_gpu)
                       .count() /
                   1e6
            << " seconds " << std::endl;

  /* std::cout << " A =" << std::endl;
  print_matrix(A, row_A, col_A);
  std::cout << " B =" << std::endl;
  print_matrix(B, row_B, col_B);
  std::cout << " C_cpu =" << std::endl;
  print_matrix(C_cpu, row_A, col_B);
  std::cout << " C_gpu =" << std::endl;
  print_matrix(C_gpu, row_A, col_B); */

  error_check(C_cpu, C_gpu, row_A, col_B);

  std::cout << " --------------------------------- " << std::endl;
  std::cout << " ---- Error checking passed!! ---- " << std::endl;
  std::cout << " --------------------------------- " << std::endl;

  // free the variables
  free(A);
  free(B);
  free(C_gpu);
  free(C_cpu);

  return 0;
}
