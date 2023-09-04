
#include <iostream>

/* ---------------------------------------------------------------- */
// Error handling
void cudaErrorCheck(cudaError_t error, bool abort = true) {
  if (error != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__,
           __LINE__);
    if (abort)
      exit(EXIT_FAILURE);
  }
}

/* ---------------------------------------------------------------- */
void printKernelData(dim3 blockDim, dim3 gridDim) {
  std::cout << "blockDim(x, y, z) : (" << blockDim.x << ", " << blockDim.y
            << ", " << blockDim.z << ") " << std::endl;
  std::cout << "gridDim(x, y, z) : (" << gridDim.x << ", " << gridDim.y << ", "
            << gridDim.z << ") " << std::endl;
}

/* ---------------------------------------------------------------- */
// Initialize the matrices A and B randomly between -50 to 50
void initialize_data(int *A, int *B, int *C_cpu, int *C_gpu, const size_t row_A,
                     const size_t row_B, const size_t col_B) {

  // Initialize to a random seed
  // srand((unsigned)time(NULL));
  srand(1);
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

/* ---------------------------------------------------------------- */
// Checking error for the gpu and cpu code
void error_check(const int *C_h, const int *C_d, const size_t row_A,
                 const size_t col_B) {
  const double tolerance = 1e-4;
  for (size_t i = 0; i < row_A; i++) {
    for (size_t j = 0; j < col_B; j++) {
      if (abs(C_h[i * col_B + j] - C_d[i * col_B + j]) > tolerance) {
        std::cout << "Error C_cpu(" << i << ", " << j
                  << ") = " << C_h[i * col_B + j] << " and C_gpu(" << i << ", "
                  << j << ") = " << C_d[i * col_B + j] << std::endl;
        exit(1);
      }
    }
  }
}

/* ---------------------------------------------------------------- */
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
