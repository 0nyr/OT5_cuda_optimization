#include <iostream>

// Contract : h_array must be already allocated
template <typename T>
T* allocateDeviceArray(T* h_array, size_t size) {
  int * d_array; // device (GPU)
  cudaError_t err = cudaMalloc((int **) &d_array, size*sizeof(int));
  if (err != cudaSuccess) {
      printf(
          "%s in %s at line %d\n", cudaGetErrorString(err),
          __FILE__,
          __LINE__
      );
      exit(EXIT_FAILURE);
  }
  return d_array;
}

