# Cuda Projects
Following are a collection of some of my cuda projects.
### 1. DeviceProperties
- A basic code to generate the device properties of the gpu present, so you can make a more informed choice of the maximum threads per block, shared memory size or maximum registers per thread and various other things.
### 2. VectorAddition
- "Hello world" equivalent of cuda program, to test if cuda is working and your basic understanding of kernel and launching with blocks and threads.
- However, it doesnt offer any significant improvement over cpu computation becuause the overhead of copying memory from device to host and viceversa is much more than the simple addition.
### 3. ConvertColorToGray
  - Similar to vector addition but in this, I use open cv to get the data from a color image and convert to gray by using a simple saxpy type of computation.
### 4. Multiplication
  - In this project, I show the comparision of CPU vs GPU for matrix multiplication. I consider, two forms of the problem where first one is the direct approach and second one is where
  I use the tiled approach which takes advantage of the shared memory and does the job even faster.
  - Computation time
  - Square Matrices
    | Matrix dimensions | CPU (seconds) | GPU Basic (seconds)  | GPU Tiled (seconds) |
    | :----: | :----:| :----: | :----:|
    |16 X 16 | 9*10<sup>-4</sup> | 2.6*10<sup> -4 </sup>| 2.3*10<sup> -4 </sup>|
    |64 X 64 | 6.2*10<sup>-4</sup> |3.1*10<sup>-4</sup> |2.4*10<sup>-4</sup> |
    |512 X 512 | 3.45*10<sup>-1</sup> | 2.8 * 10 <sup> -3</sup> | 1.6* 10<sup>-3 </sup> |
    |2048 X 2048 |1.49*10<sup>2</sup>  | 1.32 *10<sup> -1 </sup> |5.7*10<sup>-2 </sup> |

  - Rectangular Matrices
    | A dim| B dim |C dim| CPU (seconds)  | GPU Basic (seconds)  | GPU Tiled (seconds)  |
     | :----: | :----:| :----: | :----:| :----: | :----: |
    |64 X 128 | 128 X 64 | 64 X 64| 1.2*10<sup>-3</sup> |3.4*10<sup>-4</sup> | 2.7*10<sup> -4</sup> |
    |512 X 256 | 256 X 512 |512 X 512 |1.7 * 10<sup> -1</sup> |1.6 * 10<sup> -3 </sup> |  9.2*10<sup>-4</sup> |
    |1024 X 2048| 2048 x 1024|1024 X 1024| 2.85 *10<sup>1</sup> | 4.25 *10 <sup> -2 </sup> | 1.65 * 10 <sup> -2</sup> |
    | 1024 X 2048| 2048 X 728| 1024 X 728| 3.78 |2.2 * 10<sup>-2</sup> |1.1*10 <sup> -2</sup> |

### 4. Future Projects
 - Merging
 - Reduce
 - Sorting
 - Parallel Histogram
 - Suffix Array and LSP calculation
 - Suffix Array sorting algorithm
