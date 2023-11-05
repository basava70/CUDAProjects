#ifndef __UTILITY_H__
#define __UTILITY_H__

void cudaErrorCheck(cudaError_t error, bool abort = true);

void printKernelData(dim3 blockDim, dim3 gridDim);

void initialize_data(int *A, int *B, int *C_cpu, int *C_gpu, const size_t row_A,
                     const size_t row_B, const size_t col_B);

void error_check(const int *C_h, const int *C_d, const size_t row_A,
                 const size_t col_B);

void print_matrix(const int *A, const size_t row_A, const size_t col_A);

void gpuMatrixMultiplication(const int *A, const int *B, int *C,
                             const size_t row_A, const size_t row_B,
                             const size_t col_B);

void gpuTiledMatrixMultiplication(const int *A, const int *B, int *C,
                                  const size_t row_A, const size_t row_B,
                                  const size_t col_B);

void cpuMatrixMultiplication(const int *A, const int *B, int *C,
                             const size_t row_A, const size_t row_B,
                             const size_t col_B);

#endif
