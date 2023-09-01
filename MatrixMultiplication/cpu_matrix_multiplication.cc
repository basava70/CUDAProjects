
#include <iostream>

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
