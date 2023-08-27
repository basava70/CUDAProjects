#include <cuda_runtime.h>
#include <iostream>

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, i);

    std::cout << "Device Number: " << i << std::endl;
    std::cout << " Device Name: " << prop.name << std::endl;
    std::cout << " Compute Capability: " << prop.major << "." << prop.minor
              << std::endl;
    std::cout << " Total Global Memory (bytes): " << prop.totalGlobalMem
              << std::endl;
    std::cout << " Max Threads per Block: " << prop.maxThreadsPerBlock
              << std::endl;
    std::cout << " Max Blocks per MultiProcessor: "
              << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << " Memory Clock Rate (kHz): " << prop.memoryClockRate
              << std::endl;
    std::cout << " Memory Bus Width (bits): " << prop.memoryBusWidth
              << std::endl;
    std::cout << " L2 Cache Size (bytes): " << prop.l2CacheSize << std::endl;
  }

  return 0;
}
