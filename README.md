# Cuda Projects
Following are a collection of some of my cuda projects.
### 1. DeviceProperties
- A basic code to generate the device properties of the gpu present, so you can make a more informed choice of the maximum threads per block, shared memory size or maximum registers per thread and various other things.
### 2. VectorAddition
- "Hello world" equivalent of cuda program, to test if cuda is working and your basic understanding of kernel and launching with blocks and threads.
- However, it doesnt offer any significant improvement over cpu computation becuause the overhead of copying memory from device to host and viceversa is much more than the simple addition.
### 3. ConvertColorToGray
  - Similar to vector addition but in this, I use open cv to get the data from a color image and convert to gray by using a simple saxpy type of computation.
### 4. Future Projects
 - Multiplication
 - Multiplication with tiles and shared memory
 - Merging
 - Reduce
 - Sorting
 - Parallel Histogram
 - Suffix Array and LSP calculation
 - Suffix Array sorting algorithm
