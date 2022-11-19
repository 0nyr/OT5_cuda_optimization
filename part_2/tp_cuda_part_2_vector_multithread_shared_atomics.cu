/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

#define CHECK_ERROR

#include "utils.hpp"

using namespace std;

void checkSizes(long long &N, long long &M, long long &S, int &nrepeat) {
  // If S is undefined and N or M is undefined, set S to 2^22 or the bigger of N and M.
  if ( S == -1 && ( N == -1 || M == -1 ) ) {
    S = pow( 2, 22 );
    if ( S < N ) S = N;
    if ( S < M ) S = M;
  }

  // If S is undefined and both N and M are defined, set S = N * M.
  if ( S == -1 ) S = N * M;

  // If both N and M are undefined, fix row length to the smaller of S and 2^10 = 1024.
  if ( N == -1 && M == -1 ) {
    if ( S > 1024 ) {
      M = 1024;
    }
    else {
      M = S;
    }
  }

  // If only M is undefined, set it.
  if ( M == -1 ) M = S / N;

  // If N is undefined, set it.
  if ( N == -1 ) N = S / M;

  printf( "  Total size S = %lld N = %lld M = %lld\n", S, N, M );

  // Check sizes.
  if ( ( S < 0 ) || ( N < 0 ) || ( M < 0 ) || ( nrepeat < 0 ) ) {
    printf( "  Sizes must be greater than 0.\n" );
    exit( 1 );
  }

  if ( ( N * M ) != S ) {
    printf( "  N * M != S\n" );
    exit( 1 );
  }
}




__global__ void computeVectorOperation(
  unsigned long long N,
  unsigned long long M,
  int* d_x,
  int* d_y,
  int* d_A,
  unsigned long long* d_results,
  int nbThreads
) {
  // WARN: beware of thinking it is shared memory
  extern __shared__ long long array[]; // size=max(N,M)
  __shared__ int cellsPerThread;
  __shared__ unsigned long long tmpSum;

  if(threadIdx.x == 0) {
    cellsPerThread = (M/nbThreads) + 1;
  }
  array[threadIdx.x] = 0;
  __syncthreads();

  // do row(A)*X
  for (size_t i = threadIdx.x*cellsPerThread; i < threadIdx.x*cellsPerThread + cellsPerThread; i++)
  {
    if (i < M) {
      array[threadIdx.x] += d_A[blockIdx.x*M + i]*d_x[i];
    }
  }
  
  // inner block reduction step
  __syncthreads();
  for (size_t s = 1; s < blockDim.x; s *= 2) {
      int index = 2*s*threadIdx.x;

      if(index < blockDim.x && index < M) {
          array[index] += array[index + s];
      }
      __syncthreads();
  }

  if(threadIdx.x == 0) {
    cellsPerThread = ((N/nbThreads) + 1);
    tmpSum = array[0]; // keep original value
  }
  array[threadIdx.x] = 0; // reset array
  __syncthreads();

  // do tmpSum*Y
  for (size_t i = threadIdx.x*cellsPerThread; i < threadIdx.x*cellsPerThread + cellsPerThread; i++)
  {
    if (i < N) {
      array[threadIdx.x] += tmpSum*d_y[i];
    }
  }

  // inner block reduction step
  __syncthreads();
  for (size_t s = 1; s < blockDim.x; s *= 2) {
      int index = 2*s*threadIdx.x;

      if(index < blockDim.x && index < N) {
          array[index] += array[index + s];
      }
      __syncthreads();
  }

  // save result
  if(threadIdx.x == 0) {
    d_results[blockIdx.x] = array[0];
  }

}


int main( int argc, char* argv[] )
{
  // print file name
  cout << "File: " << __FILE__ << endl;

  long long N = -1;       // number of rows 2^12
  long long M = -1;       // number of columns 2^10
  long long S = -1;       // total size 2^22
  int nrepeat = 10;       // number of repeats of the test
  int nbThreads = 1;      // number of threads per block

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-Rows" ) == 0 ) ) {
      N = pow( 2, atoi( argv[ ++i ] ) );
      printf( "  User N is %lld\n", N );
    }
    else if ( ( strcmp( argv[ i ], "-M" ) == 0 ) || ( strcmp( argv[ i ], "-Columns" ) == 0 ) ) {
      M = pow( 2, atof( argv[ ++i ] ) );
      printf( "  User M is %lld\n", M );
    }
    else if ( ( strcmp( argv[ i ], "-S" ) == 0 ) || ( strcmp( argv[ i ], "-Size" ) == 0 ) ) {
      S = pow( 2, atof( argv[ ++i ] ) );
      printf( "  User S is %lld\n", S );
    }
    else if ( ( strcmp( argv[ i ], "-T" ) == 0 ) || ( strcmp( argv[ i ], "-Threads" ) == 0 ) ) {
      nbThreads = atoi( argv[ ++i ] );
      printf( "  User nbThreads is %lld\n", nbThreads );
    }
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      nrepeat = atoi( argv[ ++i ] );
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  y^T*A*x Options:\n" );
      printf( "  -Rows (-N) <int>:      exponent num, determines number of rows 2^num (default: 2^12 = 4096)\n" );
      printf( "  -Columns (-M) <int>:   exponent num, determines number of columns 2^num (default: 2^10 = 1024)\n" );
      printf( "  -Size (-S) <int>:      exponent num, determines total matrix size 2^num (default: 2^22 = 4096*1024 )\n" );
      printf( "  -Threads (-T) <int>:   number of threads per block (default : 1)\n" );
      printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }

  // Check sizes.
  checkSizes( N, M, S, nrepeat );

  // Allocate x,y,A
  // Initialize y vector to 1.
  // Initialize x vector to 1.
  // Initialize A matrix, you can use a 1D index if you want a flat structure (i.e. a 1D array) e.g. j*M+i is the same than [j][i]
  
  // Allocating the vectors and matrix on Host
  int* h_x = (int *) malloc(M* sizeof(int));
  for (size_t i = 0; i < M; i++)
  {
    h_x[i] = 1;
  }
  int* h_y = (int *) malloc(N* sizeof(int));
  for (size_t i = 0; i < N; i++)
  {
    h_y[i] = 1;
  }
  // A is a linearized matrix
  int* h_A = (int *) malloc(M*N*sizeof(int));
  for(size_t i = 0; i < N; i++) {
    for(size_t j = 0; j < M; j++) {
      h_A[i*M + j] = 1;
    }
  }

  // Preparing ground for results
  unsigned long long* h_results = (unsigned long long *) malloc(N*sizeof(unsigned long long)); // host (CPU)
  unsigned long long* d_results = allocateDeviceArray<unsigned long long>(N);

  // Giving the vectors to device
  int* d_x = allocateDeviceArray<int>(M);
  int* d_y = allocateDeviceArray<int>(N);
  int* d_A = allocateDeviceArray<int>(M*N);
  cudaMemcpy(d_x, h_x, M*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, h_A, N*M*sizeof(int), cudaMemcpyHostToDevice);

  // Timer products.
  struct timeval begin, end;

  gettimeofday( &begin, NULL );

  // WARN: perf evaluation = DON'T PARALLEL !!!
  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
    computeVectorOperation<<<N, nbThreads, nbThreads*sizeof(unsigned long long)>>>(
      N, M, d_x, d_y, d_A, d_results, nbThreads
    );

    // get back result from device
    cudaMemcpy(h_results, d_results, N*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    unsigned long long result = *h_results; // 


    // Output result.
    if ( repeat == ( nrepeat - 1 ) ) {
      printf( "  Computed result for %lld x %lld is %lld\n", N, M, result);
    }

    // chech results
    const unsigned long long solution = N*M;
    for(int i = 0; i < N; i++) {
      if ( h_results[i] != solution ) {
        #ifdef CHECK_ERROR
        printf( "[%d]  Error: result( %lld ) != solution( %lld )\n", repeat, result, solution);
        #endif
      }
    }
    
  }

  gettimeofday( &end, NULL );

  // Calculate time.
  //double time = timer.seconds();
  double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  // The x vector (of length M) is read N times.
  // The y vector (of length N) is read once.
  // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
  double Gbytes = 1.0e-9 * double( sizeof(double) * ( M + M * N + N ) );

  // Print results (problem size, time and bandwidth in GB/s).
  printf( "  N( %lld ) M( %lld ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
          N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time );

  free(h_results);
  cudaFree(d_results);
  free(h_x);
  cudaFree(d_x);
  free(h_y);
  cudaFree(d_y);
  free(h_A);
  cudaFree(d_A);

  // output to file 
  ofstream myfile("stats.csv", ios::app);
  if (myfile.is_open())
  {
      myfile << "multithread_shared" << "," 
        << S << ","
        << T << ","
        << std::setprecision(std::numeric_limits<double>::digits10) << time
        << endl;
      myfile.close();
  }
  else cerr<<"Unable to open file";

  return 0;
}




