
/**
 * Blurring a given image using CUDA.
 *Created on August 27th 2023.
 *
 **/

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <string>

unsigned char *image_rgb_h;
#define CHANNELS 3 // number of channels 3 for r g b

// Error handling
inline void cudaErrorCheck(cudaError_t error, bool abort = true) {
  if (error != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__,
           __LINE__);
    if (abort)
      exit(EXIT_FAILURE);
  }
}

// Printing the kernel data
void printKernelData(dim3 blockDim, dim3 gridDim) {
  std::cout << "blockDim(x, y, z) : (" << blockDim.x << ", " << blockDim.y
            << ", " << blockDim.z << ") " << std::endl;
  std::cout << "gridDim(x, y, z) : (" << gridDim.x << ", " << gridDim.y << ", "
            << gridDim.z << ") " << std::endl;
}

// Function to load the input image
void loadImage(std::string input_file, size_t *rows, size_t *cols) {

  // Create a temporary OpenCV Mat variable to load the input image
  cv::Mat image;
  image = cv::imread(input_file.c_str(), cv::IMREAD_COLOR);
  if (image.empty() || !image.data) {
    std::cerr << "Unable to load image " << input_file << std::endl;
    exit(1);
  }

  *rows = image.rows;
  *cols = image.cols;

  // Allocate memory for host rgb image variable
  image_rgb_h =
      (unsigned char *)malloc(*rows * *cols * sizeof(unsigned char) * 3);

  // Create a tmp variable to copy data from cv::Mat
  unsigned char *rgb_image = (unsigned char *)image.data;

  // Populate host's rgb data array
  size_t iter = 0;
  for (iter = 0; iter < *rows * *cols * 3; iter++) {
    image_rgb_h[iter] = rgb_image[iter];
  }
}

// Seperate the 3 channels of the color image into 3 individual channels
// using a gpu kernel
__global__ void seperateColors(unsigned char *red_d, unsigned char *green_d,
                               unsigned char *blue_d,
                               unsigned char *image_rgb_d, size_t rows,
                               size_t cols) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t idy = blockDim.y * blockIdx.y + threadIdx.y;

  if (idx <= rows && idy <= cols) {
    size_t offset = idy * cols + idx;

    red_d[offset] = image_rgb_d[offset * 3 + 0];
    green_d[offset] = image_rgb_d[offset * 3 + 1];
    blue_d[offset] = image_rgb_d[offset * 3 + 2];
  }
}

// Once we get the blurred individual channels, we combine them together to
// get a full color image
__global__ void combineColors(unsigned char *red_d, unsigned char *green_d,
                              unsigned char *blue_d,
                              unsigned char *output_image, const size_t rows,
                              const size_t cols) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t idy = blockDim.y * blockIdx.y + threadIdx.y;

  if (idx <= rows && idy <= cols) {
    size_t offset = idy * cols + idx;
    output_image[offset * 3 + 0] = red_d[offset];
    output_image[offset * 3 + 1] = green_d[offset];
    output_image[offset * 3 + 2] = blue_d[offset];
  }
}

// GPU Kernel to do the blur of the given channel
// We choose a filter of size "filter_size" and add
// all the pixels and divide by the number of pixels

// Ex: filter_size = 3, we have
// blurrer_channel(i,j) =   sum of all the elements in the
// 3 * 3 matrix with (i,j) at the center

// ------------------------------------
//  (i-1,j-1)  (i, j-1)  (i+1, j-1)
//   (i-1,j)    (i, j)    (i+1, j)
//  (i-1,j+1)  (i, j+1)  (i+1, j+1)
// ------------------------------------

__global__ void blurKernel(unsigned char *input_channel,
                           unsigned char *output_channel, const int filter_size,
                           const size_t rows, const size_t cols) {

  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t idy = blockDim.y * blockIdx.y + threadIdx.y;

  const int filter_length = filter_size / 2;

  if (idx <= rows && idy <= cols) {
    int total_pixels_in_filter = 0;
    int total_pixel_value_in_filter = 0;

    // looping over the filter matrix of size filter_size * filter_size
    for (int row_index = -filter_length; row_index < filter_length + 1;
         row_index++) {
      for (int col_index = -filter_length; col_index < filter_length + 1;
           col_index++) {
        int filter_row = row_index + idx;
        int filter_col = col_index + idy;

        // we loop over only the indices that are in the original image
        // accouting for the boundary terms
        // total_pixels_in_filter keeps count of how many pixels
        // of the kernel are in the image
        // For the top left boundary, we have (i,j) = (0,0)
        // the only valid pixels at (0,0) are 4, i.e.,
        // (0,0) (1,0)
        // (0,1) (1,1)
        // In this case, we would sum over all the four pixels and divide by 4
        if (filter_row >= 0 && filter_row <= rows && filter_col >= 0 &&
            filter_col <= cols) {
          total_pixels_in_filter++;
          total_pixel_value_in_filter +=
              input_channel[filter_col * cols + filter_row];
        }
      }
    }
    output_channel[idy * cols + idx] =
        (unsigned char)(total_pixel_value_in_filter / total_pixels_in_filter);
  }
}

// Utility function which facilitates all the things needed for
// blur using cuda. Creating host and device variables for each channel,
// and allocating memory and calling the kernel.
void gpuBlur(unsigned char *blurred_image_h, const int filter_size,
             const size_t rows, const size_t cols) {
  unsigned char *red_d;
  unsigned char *green_d;
  unsigned char *blue_d;

  unsigned char *blurred_red_d;
  unsigned char *blurred_green_d;
  unsigned char *blurred_blue_d;

  unsigned char *image_rgb_d;

  const size_t total_pixels = rows * cols;

  // We only need device variables for 3 channels; no need of host variables for
  // 3 channels
  cudaErrorCheck(
      cudaMalloc((void **)&red_d, total_pixels * sizeof(unsigned char)));
  cudaErrorCheck(
      cudaMalloc((void **)&green_d, total_pixels * sizeof(unsigned char)));
  cudaErrorCheck(
      cudaMalloc((void **)&blue_d, total_pixels * sizeof(unsigned char)));

  // Device variables for blurred channels
  cudaErrorCheck(cudaMalloc((void **)&blurred_red_d,
                            total_pixels * sizeof(unsigned char)));
  cudaErrorCheck(cudaMalloc((void **)&blurred_green_d,
                            total_pixels * sizeof(unsigned char)));
  cudaErrorCheck(cudaMalloc((void **)&blurred_blue_d,
                            total_pixels * sizeof(unsigned char)));

  // Device variable for the input image
  cudaErrorCheck(cudaMalloc((void **)&image_rgb_d,
                            total_pixels * sizeof(unsigned char) * CHANNELS));

  // copying image variable from host to device
  cudaErrorCheck(cudaMemcpy(image_rgb_d, image_rgb_h,
                            total_pixels * sizeof(unsigned char) * CHANNELS,
                            cudaMemcpyHostToDevice));

  // Specify the variables needed for GPU kernel
  const size_t threads_per_block = 16;
  const dim3 block_dimensions(threads_per_block, threads_per_block);
  const dim3 grid_dimesions(ceil(float(rows) / threads_per_block),
                            ceil(float(cols) / threads_per_block));

  // Calling the GPU kernel to seperate all the three channels parallely
  seperateColors<<<grid_dimesions, block_dimensions>>>(red_d, green_d, blue_d,
                                                       image_rgb_d, rows, cols);

  // We call blur kernel for each channel seperately
  blurKernel<<<grid_dimesions, block_dimensions>>>(red_d, blurred_red_d,
                                                   filter_size, rows, cols);
  blurKernel<<<grid_dimesions, block_dimensions>>>(green_d, blurred_green_d,
                                                   filter_size, rows, cols);
  blurKernel<<<grid_dimesions, block_dimensions>>>(blue_d, blurred_blue_d,
                                                   filter_size, rows, cols);

  // Once we get the blurred channels, we combine them together into
  // the device variable
  combineColors<<<grid_dimesions, block_dimensions>>>(
      blurred_red_d, blurred_green_d, blurred_blue_d, image_rgb_d, rows, cols);

  // copy the output image to host
  cudaErrorCheck(cudaMemcpy(blurred_image_h, image_rgb_d,
                            total_pixels * sizeof(unsigned char) * CHANNELS,
                            cudaMemcpyDeviceToHost));

  // Free memory from the device
  cudaFree(red_d);
  cudaFree(green_d);
  cudaFree(blue_d);
  cudaFree(blurred_red_d);
  cudaFree(blurred_green_d);
  cudaFree(blurred_blue_d);
  cudaFree(image_rgb_d);
}

// Function to output the image again
void outputImage(unsigned char *image, const std::string output_file,
                 const size_t rows, const size_t cols) {

  cv::Mat output(rows, cols, CV_8UC3, (void *)image);
  cv::imwrite(output_file.c_str(), output);
  cv::imshow("Blurred Image", output);
  cv::waitKey(0);
}

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
  loadImage(input_file, &rows, &cols);

  size_t total_pixels = rows * cols;

  std::cout << "--------------------------" << std::endl;
  std::cout << " Dimensions of the input " << std::endl;
  std::cout << " rows : " << rows << std::endl;
  std::cout << " cols : " << cols << std::endl;
  std::cout << " total_pixels : " << total_pixels << std::endl;
  std::cout << "--------------------------" << std::endl;

  // show the original image
  cv::Mat output(rows, cols, CV_8UC3, (void *)image_rgb_h);
  cv::imwrite(output_file.c_str(), output);
  cv::imshow("Input Image", output);

  unsigned char *blurred_image_h;
  blurred_image_h =
      (unsigned char *)malloc(rows * cols * sizeof(unsigned char) * CHANNELS);

  // Choosing filter length to be 3 ( choose only odd numbers for now)
  const int filter_size = 5;
  gpuBlur(blurred_image_h, filter_size, rows, cols);

  outputImage(blurred_image_h, output_file, rows, cols);

  // free cpu memory
  free(image_rgb_h);
  free(blurred_image_h);

  return 0;
}
