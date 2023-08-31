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
void initialize_data(int *A, int *B, int *C_cpu, int *C_gpu, const size_t row_A,
                     const size_t row_B, const size_t col_B) {

  // Initialize to a random seed
  // srand((unsigned)time(NULL));
  srand(1);
  std::cout << "Started initialization " << std::endl;
  for (size_t i = 0; i < row_A; i++) {
    for (size_t j = 0; j < row_B; j++) {
      A[i * row_B + j] = (rand() % 10);
    }
  }

  for (size_t i = 0; i < row_B; i++) {
    for (size_t j = 0; j < col_B; j++) {
      B[i * col_B + j] = (rand() % 10);
    }
  }

  for (size_t i = 0; i < row_A; i++) {
    for (size_t j = 0; j < col_B; j++) {
      C_cpu[i * col_B + j] = 0;
      C_gpu[i * col_B + j] = 0;
    }
  }
}

void cpuMatrixMultiplication(const int *A, const int *B, int *C,
                             const size_t row_A, const size_t row_B,
                             const size_t col_B) {
  for (size_t i = 0; i < row_A; i++) {
    for (size_t j = 0; j < col_B; j++) {
      int value = 0;
      for (size_t k = 0; k < row_B; k++) {
        value += A[i * row_B + k] * B[k * col_B + j];
      }
      C[i * col_B + j] = value;
    }
  }
}

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

void gpuMatrixMultiplication(const int *A, const int *B, int *C,
                             const size_t row_A, const size_t row_B,
                             const size_t col_B) {

  // Creating device variables for all the three matrices
  int *a_d, *b_d, *c_d;

  // Allocating memory for device variables
  cudaErrorCheck(cudaMalloc((void **)&a_d, row_A * row_B * sizeof(int)));
  cudaErrorCheck(cudaMalloc((void **)&b_d, row_B * col_B * sizeof(int)));
  cudaErrorCheck(cudaMalloc((void **)&c_d, row_A * col_B * sizeof(int)));

  cudaErrorCheck(
      cudaMemcpy(a_d, A, row_A * row_B * sizeof(int), cudaMemcpyHostToDevice));
  cudaErrorCheck(
      cudaMemcpy(b_d, B, row_B * col_B * sizeof(int), cudaMemcpyHostToDevice));

  const size_t threads_per_block = 16;
  dim3 block_dimension(threads_per_block, threads_per_block);
  dim3 grid_dimension(ceil(float(row_A) / threads_per_block),
                      ceil(float(col_B) / threads_per_block));

  printKernelData(block_dimension, grid_dimension);

  gpuKernel<<<grid_dimension, block_dimension>>>(a_d, b_d, c_d, row_A, row_B,
                                                 col_B);

  cudaErrorCheck(
      cudaMemcpy(C, c_d, row_A * col_B * sizeof(int), cudaMemcpyDeviceToHost));

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
}

void error_check(const int *C_h, const int *C_d, const size_t row_A,
                 const size_t col_B) {
  const double tolerance = 1e-4;
  for (size_t i = 0; i < row_A; i++) {
    for (size_t j = 0; j < col_B; j++) {
      if (C_h[i * col_B + j] - C_d[i * col_B + j] > tolerance) {
        std::cout << "Error C_cpu(" << i << ", " << j
                  << ") = " << C_h[i * col_B + j] << " and C_gpu(" << i << ", "
                  << j << ") = " << C_d[i * col_B + j] << std::endl;
        exit(1);
      }
    }
  }
}

void print_matrix(const int *A, const size_t row_A, const size_t col_A) {
  for (size_t i = 0; i < row_A; i++) {
    std::cout << "|";
    for (size_t j = 0; j < col_A; j++) {
      std::cout << " " << A[i * col_A + j];
    }
    std::cout << " |";
    std::cout << std::endl;
  }
}

int main() {

  const size_t row_A = 512, row_B = 1024, col_A = 1024, col_B = 512;
  int A[row_A * col_A], B[row_B * col_B], C_cpu[row_A * col_B];

  if (row_B != col_A) {
    std::cout << " row_b != col_A " << std::endl;
    exit(1);
  }

  int *C_gpu;
  C_gpu = (int *)malloc(row_A * col_B * sizeof(int));

  std::cout << " /***********************************************/ "
            << std::endl;
  std::cout << "  *************Matrix multiplication*************  "
            << std::endl;
  std::cout << "  /***********************************************/ "
            << std::endl;
  std::cout << " row_A : " << row_A << " col_A : " << col_A << std::endl;
  std::cout << " row_B : " << row_B << " col_B : " << col_B << std::endl;
  std::cout << " row_C : " << row_A << " col_C : " << col_B << std::endl;

  initialize_data(&A[0], &B[0], &C_cpu[0], C_gpu, row_A, row_B, col_B);

  auto begin_gpu = std::chrono::high_resolution_clock::now();
  gpuMatrixMultiplication(&A[0], &B[0], C_gpu, row_A, row_B, col_B);
  auto end_gpu = std::chrono::high_resolution_clock::now();

  std::cout << "Time elapsed for gpu is: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_gpu -
                                                                     begin_gpu)
                   .count()
            << " micro seconds " << std::endl;

  auto begin_cpu = std::chrono::high_resolution_clock::now();
  cpuMatrixMultiplication(&A[0], &B[0], &C_cpu[0], row_A, row_B, col_B);
  auto end_cpu = std::chrono::high_resolution_clock::now();

  std::cout << "Time elapsed for cpu is: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu -
                                                                     begin_cpu)
                   .count()
            << " micro seconds " << std::endl;

  /* std::cout << " A =" << std::endl;
  print_matrix(&A[0], row_A, col_A);
  std::cout << " B =" << std::endl;
  print_matrix(&B[0], row_B, col_B);
  std::cout << " C_cpu =" << std::endl;
  print_matrix(&C_cpu[0], row_A, col_B);
  std::cout << " C_gpu =" << std::endl;
  print_matrix(C_gpu, row_A, col_B); */

  error_check(&C_cpu[0], &C_gpu[0], row_A, col_B);

  std::cout << " ------------------------ " << std::endl;
  std::cout << " ------- Success -------- " << std::endl;
  std::cout << " ------------------------ " << std::endl;

  free(C_gpu);

  return 0;
}
