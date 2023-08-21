/**
 * Converting color image to gray
 * using cuda. Created on August 20
 *
 **/

#include <cstddef>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <string>

unsigned char *image_rgb_h;
#define CHANNELS 3 // number of channels 3 for r g b

// error handling
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

/**
 * Function to load the input image and returns the number of
 * total pixels
 **/
int loadImage(std::string input_file, size_t *rows, size_t *cols) {
  cv::Mat image;
  image = cv::imread(input_file.c_str(), cv::IMREAD_COLOR);
  if (image.empty() || !image.data) {
    std::cerr << "Unable to load image " << input_file << std::endl;
    exit(1);
  }
  *rows = image.rows;
  *cols = image.cols;

  // allocate memory for host rgb image variable
  image_rgb_h =
      (unsigned char *)malloc(*rows * *cols * sizeof(unsigned char) * 3);
  // create a tmp variable to copy data from cv::Mat
  unsigned char *rgb_image = (unsigned char *)image.data;

  // populate host's rgb data array
  size_t iter = 0;
  for (iter = 0; iter < *rows * *cols * 3; iter++) {
    image_rgb_h[iter] = rgb_image[iter];
  }

  size_t num_of_pixels = image.rows * image.cols;

  return num_of_pixels;
}

/**
 * Kernel for color changing via cuda
 **/
__global__ void convertRGBToGrayGPUKernel(unsigned char *rgb,
                                          unsigned char *gray, size_t rows,
                                          size_t cols) {

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx <= rows && idy <= cols) {
    size_t grayoffset = idy * cols + idx;
    size_t rgboffset = grayoffset * CHANNELS;

    unsigned char r = rgb[rgboffset + 0];
    unsigned char g = rgb[rgboffset + 1];
    unsigned char b = rgb[rgboffset + 2];

    gray[grayoffset] = r * 0.299f + g * 0.587f + b * 0.114f;
  }
}

/**
 * Utility for the gpu of color conversion to gray.
 * Creating and allocating memory in device.
 * Copying the memory to device from host
 * and send it to the kernel and copy it back to
 * host. Free the memory after.
 **/
void convertRGBToGrayGPU(unsigned char *image_rgb_h,
                         unsigned char *image_gray_h, const size_t total_pixels,
                         const size_t rows, const size_t cols,
                         bool verbose = false) {

  unsigned char *image_gray_d;
  unsigned char *image_rgb_d;

  // allocating the memory for device variables
  cudaErrorCheck(cudaMalloc((void **)&image_rgb_d,
                            total_pixels * sizeof(unsigned char) * CHANNELS));
  cudaErrorCheck(
      cudaMalloc((void **)&image_gray_d, total_pixels * sizeof(unsigned char)));
  cudaErrorCheck(
      cudaMemset(image_gray_d, 0, total_pixels * sizeof(unsigned char)));

  // copy data from host to device
  cudaErrorCheck(cudaMemcpy(image_rgb_d, image_rgb_h,
                            total_pixels * sizeof(unsigned char) * CHANNELS,
                            cudaMemcpyHostToDevice));

  size_t threads_per_block = 16;
  dim3 blockDim(16, 16);
  dim3 gridDim(ceil(float(rows) / threads_per_block),
               ceil(float(cols) / threads_per_block));

  if (verbose == true)
    printKernelData(blockDim, gridDim);

  convertRGBToGrayGPUKernel<<<gridDim, blockDim>>>(image_rgb_d, image_gray_d,
                                                   rows, cols);

  cudaErrorCheck(cudaMemcpy(image_gray_h, image_gray_d,
                            total_pixels * sizeof(unsigned char),
                            cudaMemcpyDeviceToHost));

  // free memory in gpu
  cudaFree(image_rgb_d);
  cudaFree(image_gray_d);
}

void outputImage(unsigned char *gray, const std::string output_file,
                 const size_t rows, const size_t cols) {

  cv::Mat output(rows, cols, CV_8UC1, (void *)gray);
  cv::imwrite(output_file.c_str(), output);
  cv::imshow("GrayImage", output);
  cv::waitKey(0);
}

// Global_variables

int main(int argc, char **argv) {
  // declaring input and output file variables
  std::string input_file;
  std::string output_file;

  switch (argc) {
  case 2:
    input_file = std::string(argv[1]);
    output_file = "output.jpg";
    break;

  case 3:
    input_file = std::string(argv[1]);
    output_file = std::string(argv[2]);
    break;

  default:
    std::cerr << "Usage: <executable> input_file output_file" << std::endl;
    exit(1);
  }

  // dimension variables in the image
  size_t rows;
  size_t cols;

  // lets read the input file
  size_t total_pixels = loadImage(input_file, &rows, &cols);

  std::cout << "--------------------------" << std::endl;
  std::cout << " Dimensions of the input " << std::endl;
  std::cout << " rows : " << rows << std::endl;
  std::cout << " cols : " << cols << std::endl;
  std::cout << " total_pixels : " << total_pixels << std::endl;
  std::cout << "--------------------------" << std::endl;

  // show the original image
  cv::Mat output(rows, cols, CV_8UC3, (void *)image_rgb_h);
  cv::imwrite(output_file.c_str(), output);
  cv::imshow("rgb_h", output);

  // create host image variable
  unsigned char *image_gray_h;
  image_gray_h =
      (unsigned char *)malloc(total_pixels * sizeof(unsigned char *));

  convertRGBToGrayGPU(image_rgb_h, image_gray_h, total_pixels, rows, cols,
                      true);

  outputImage(image_gray_h, output_file, rows, cols);

  // free cpu memory
  free(image_rgb_h);
  free(image_gray_h);

  return 0;
}
