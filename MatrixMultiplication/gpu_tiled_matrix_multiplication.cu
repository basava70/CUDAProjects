

#define TILEWIDTH 32
void cudaErrorCheck(cudaError_t error, bool abort = true);

void printKernelData(dim3 blockDim, dim3 gridDim);

/**
  * Consider the example with TILEWIDTH 2 and A, B are square matrices of
  * size 4 * 4. We consider the blocksize equal to TILEWIDTH. In this example
  * we have 4 blocks Block(0,0), Block(0,1), Block(0,2) and Block(0,3).
  * In each block, we create shared tiles of size TILEWIDTH * TILEWIDTH =
  * (2 * 2). We divide the sum into phases = Col_B/TILEWIDTH.
  *
  * ------------------------------------------------------------------ *
  *                                         ----------------
  *                           phase 0    | | B(0,0) B(0,1) | B(0,2) B(0,3) |
  *                                      | | B(1,0) B(1,1) | B(1,2) B(1,3) |
  *                                      | ----------------                |
  *                           phase 1    |   B(2,0) B(2,1)   B(2,2) B(2,3) |
  *                                      |   B(3,0) B(3,1)   B(3,2) B(3,3) |

  *         phase 0         phase 1
  *    -----------------
  *  | | A(0,0) A(0,1) | A(0,2) A(0,3) |
  *  | | A(1,1) A(1,1) | A(1,2) A(1,3) |
  *  | -----------------               |
  *  |   A(2,2) A(2,1)   A(2,2) A(2,3) |
  *  |   A(3,3) A(3,1)   A(3,2) A(3,3) |
  *
  * ------------------------------------------------------------------ *
  *                 col_A
  * We have C(i,j) = Sum {A(i,k) * B(k,j)}
  *                 k = 0
  * For a given block, we have the starting indices are
  * idx = blockIdx.y * * TILEWIDTH + threadIdx.y
  * idy = blockIdx.x * * TILEWIDTH + threadIdx.x
  * We store the shared tiles for phase 0.
  * Ad[ty][tx] = A[idy * col_A + ty]
  * Bd[ty][tx] = B[tx * col_B + idx]
  * For phase 1, we move TILEWIDTH columns right for A and TILEWIDTH rows down
  * for B. Which pushes ty ==> ty + ph*TILEWIDTH for A and tx ==> tx +
  * ph*TILEWIDTH, changes
  * Ad[ty][tx] = A[idy * col_A + ty + phase * TILEWIDTH]
  * Bd[ty][tx] = B[(tx + phase * TILEWIDTH)* col_B + idx]
  *
  * ------------------------------------------------------------------ *
  **/

/* ---------------------------------------------------------------- */
__global__ void gpuTiledKernel(const int *A, const int *B, int *C,
                               const size_t row_A, const size_t row_B,
                               const size_t col_B) {
  __shared__ int Ad[TILEWIDTH][TILEWIDTH];
  __shared__ int Bd[TILEWIDTH][TILEWIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int idx = bx * TILEWIDTH + tx;
  int idy = by * TILEWIDTH + ty;

  int value = 0;
  // Adding boundary conditions
  for (size_t ph = 0; ph < ceil(row_B / (float)TILEWIDTH); ph++) {
    if ((idx < row_A) && ((ph * TILEWIDTH + ty) < row_B))
      Ad[tx][ty] = A[idx * row_B + ty + ph * TILEWIDTH];
    else
      Ad[tx][ty] = 0;
    if (((ph * TILEWIDTH + tx) < row_B) && (idy < col_B))
      Bd[tx][ty] = B[(tx + ph * TILEWIDTH) * col_B + idy];
    else
      Bd[tx][ty] = 0;
    __syncthreads();

    for (size_t k = 0; k < TILEWIDTH; k++) {
      value += Ad[tx][k] * Bd[k][ty];
    }
    __syncthreads();
  }
  if ((idx < row_A) && (idy < col_B))
    C[idx * col_B + idy] = value;
}

/* ---------------------------------------------------------------- */
void gpuTiledMatrixMultiplication(const int *A, const int *B, int *C,
                                  const size_t row_A, const size_t row_B,
                                  const size_t col_B) {

  // Creating device variables for all the three matrices
  int *a_d, *b_d, *c_d;

  // Allocating memory for device variables
  cudaErrorCheck(cudaMalloc((void **)&a_d, row_A * row_B * sizeof(int)));
  cudaErrorCheck(cudaMalloc((void **)&b_d, row_B * col_B * sizeof(int)));
  cudaErrorCheck(cudaMalloc((void **)&c_d, row_A * col_B * sizeof(int)));

  // Copying the two input matricws to device from host
  cudaErrorCheck(
      cudaMemcpy(a_d, A, row_A * row_B * sizeof(int), cudaMemcpyHostToDevice));
  cudaErrorCheck(
      cudaMemcpy(b_d, B, row_B * col_B * sizeof(int), cudaMemcpyHostToDevice));

  const size_t threads_per_block = TILEWIDTH;
  dim3 block_dimension(threads_per_block, threads_per_block);
  dim3 grid_dimension(ceil(float(row_A) / threads_per_block),
                      ceil(float(col_B) / threads_per_block));

  printKernelData(block_dimension, grid_dimension);

  // Call the kernel
  gpuTiledKernel<<<grid_dimension, block_dimension>>>(a_d, b_d, c_d, row_A,
                                                      row_B, col_B);

  // Copy variable back to host from device
  cudaErrorCheck(
      cudaMemcpy(C, c_d, row_A * col_B * sizeof(int), cudaMemcpyDeviceToHost));

  // Freeing the memory
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
}
