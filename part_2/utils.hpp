#include <iostream>

// Contract : h_array must be already allocated
template <typename T>
T* allocateDeviceArray(size_t size) {
  T * d_array; // device (GPU)
  cudaError_t err = cudaMalloc((T **) &d_array, size*sizeof(T));
  if (err != cudaSuccess) {
      printf("Cuda is doing wrong ! Probably due to memory shortage");
      exit(EXIT_FAILURE);
  }
  return d_array;
}