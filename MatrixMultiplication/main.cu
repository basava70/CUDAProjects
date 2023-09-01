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

#include <chrono>
#include <cstdlib>
#include <iostream>

void initialize_data(int *A, int *B, int *C_cpu, int *C_gpu, const size_t row_A,
                     const size_t row_B, const size_t col_B);

void error_check(const int *C_h, const int *C_d, const size_t row_A,
                 const size_t col_B);

void print_matrix(const int *A, const size_t row_A, const size_t col_A);

void gpuMatrixMultiplication(const int *A, const int *B, int *C,
                             const size_t row_A, const size_t row_B,
                             const size_t col_B);

void cpuMatrixMultiplication(const int *A, const int *B, int *C,
                             const size_t row_A, const size_t row_B,
                             const size_t col_B);

int main() {

  const size_t row_A = 1024, row_B = 2048, col_A = 2048, col_B = 1024;

  int *A, *B, *C_cpu, *C_gpu;
  A = (int *)malloc(row_A * col_A * sizeof(int));
  B = (int *)malloc(row_B * col_B * sizeof(int));
  C_cpu = (int *)malloc(row_A * col_B * sizeof(int));
  C_gpu = (int *)malloc(row_A * col_B * sizeof(int));

  if (row_B != col_A) {
    std::cout << " row_b != col_A " << std::endl;
    exit(1);
  }

  std::cout << " ----------------------------- " << std::endl;
  std::cout << " --- Matrix Multiplication --- " << std::endl;
  std::cout << " ----------------------------- " << std::endl;
  std::cout << " row_A : " << row_A << " col_A : " << col_A << std::endl;
  std::cout << " row_B : " << row_B << " col_B : " << col_B << std::endl;
  std::cout << " row_C : " << row_A << " col_C : " << col_B << std::endl;

  initialize_data(A, B, C_cpu, C_gpu, row_A, row_B, col_B);

  auto begin_gpu = std::chrono::high_resolution_clock::now();
  gpuMatrixMultiplication(A, B, C_gpu, row_A, row_B, col_B);
  auto end_gpu = std::chrono::high_resolution_clock::now();

  std::cout << "Time elapsed for gpu is: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_gpu -
                                                                     begin_gpu)
                       .count() /
                   1e6
            << " seconds " << std::endl;

  auto begin_cpu = std::chrono::high_resolution_clock::now();
  cpuMatrixMultiplication(A, B, C_cpu, row_A, row_B, col_B);
  auto end_cpu = std::chrono::high_resolution_clock::now();

  std::cout << "Time elapsed for cpu is: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu -
                                                                     begin_cpu)
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

  std::cout << " ------------------------ " << std::endl;
  std::cout << " ------- Success -------- " << std::endl;
  std::cout << " ------------------------ " << std::endl;

  // free the variables
  free(A);
  free(B);
  free(C_gpu);
  free(C_cpu);

  return 0;
}
