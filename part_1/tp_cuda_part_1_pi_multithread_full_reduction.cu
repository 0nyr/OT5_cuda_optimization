/*

This program will numerically compute the integral of

                  4/(1+x*x) 
				  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

// file handling
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

__global__ void precomputePiKernel(
    long num_steps, 
    float step,
    unsigned long nbComputePerThreadPerBlock,
    int threadsPerBlock, 
    float * d_sum_array
) { 
    // shared memory temporary result
    extern __shared__ float pre_array[];
    
    long long i = threadIdx.x + blockDim.x*blockIdx.x;

    // start computing first step
    float tmp_thread_sum = 0.0;
    float x;
    for (
        long long j = i*nbComputePerThreadPerBlock; 
        j < (i + 1)*nbComputePerThreadPerBlock; j++
    ) {
        if (j <= num_steps) {
            x = (j-0.5)*step;
            tmp_thread_sum = tmp_thread_sum + 4.0/(1.0+x*x);
        }
    }
    pre_array[threadIdx.x] = tmp_thread_sum;
    
    // do inner block reduction sum
    __syncthreads();
    for (size_t s = 1; s < blockDim.x; s *= 2) {
        int index = 2*s*threadIdx.x;

        if(index < blockDim.x) {
            pre_array[index] += pre_array[index + s];
        }
        __syncthreads();
    }
    
    // write result to global device memory
    if(threadIdx.x == 0) {
        d_sum_array[blockIdx.x] = pre_array[0];
    }
}

__global__ void reductionSumArray(
    float * array,
    size_t size
    ) {
    extern __shared__ float shared_mem_array[];

    long long i = threadIdx.x + blockDim.x*blockIdx.x;

    // each thread of the block participate in copying
    // the array from global mem to shared block mem.
    if (i < size) {
        shared_mem_array[threadIdx.x] = array[i];
    } else {
        shared_mem_array[threadIdx.x] = 0.0;
    }

    // inner block reduction step
    __syncthreads();
    for (size_t s = 1; s < blockDim.x; s *= 2) {
        int index = 2*s*threadIdx.x;

        if(index < blockDim.x) {
            shared_mem_array[index] += shared_mem_array[index + s];
        }
        __syncthreads();
    }

    // we will store final result inside the original array (overwriting values)
    // write result to global device memory
    if(threadIdx.x == 0) {
        array[blockIdx.x] = shared_mem_array[0];
    }
}

float computePi(
    long num_steps, 
    float step,
    int nb_blocks,
    int threadsPerBlock
) {
    // memory allocations
    float * h_sum_array = (float *) malloc(nb_blocks*sizeof(float)); // host (CPU)
    float * d_sum_array;
    cudaError_t err = cudaMalloc((float **) &d_sum_array, nb_blocks*sizeof(float));
    if (err != cudaSuccess) {
        printf(
            "%s in %s at line %d\n", cudaGetErrorString(err),
            __FILE__,
            __LINE__
        );
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_sum_array, h_sum_array, nb_blocks*sizeof(float), cudaMemcpyHostToDevice);

    // prepare computing
    unsigned long nbComputePerThreadPerBlock = num_steps / (nb_blocks * threadsPerBlock);

    // do computation in device (GPU)
    size_t deviceSharedBlockArraySize = threadsPerBlock*sizeof(float);
    precomputePiKernel<<<nb_blocks, threadsPerBlock, deviceSharedBlockArraySize>>>(
        num_steps, step, nbComputePerThreadPerBlock, threadsPerBlock, d_sum_array
    );
    cudaDeviceSynchronize(); // kernel functions are async

    size_t nb_useful_blocks = (size_t)((nb_blocks / threadsPerBlock) + 0.5); // upper round
    size_t arraySize = nb_blocks;
    while(nb_useful_blocks >= 1 ) {
        cudaDeviceSynchronize(); // kernel functions are async
        
        // perform reduction on array
        
        reductionSumArray<<<nb_useful_blocks, threadsPerBlock, deviceSharedBlockArraySize>>>(
            d_sum_array, arraySize
        );
        arraySize = nb_useful_blocks; // WARN: before updating nbUsefulBlock
        nb_useful_blocks = (size_t)((arraySize / threadsPerBlock) + 0.5); // upper round
    }

    // get back result from device
    cudaMemcpy(h_sum_array, d_sum_array, nb_blocks*sizeof(float), cudaMemcpyDeviceToHost);

    // compute sync sum
    float sync_sum = 0.0;
    for (int i = 0; i < nb_blocks; i++){
        sync_sum += h_sum_array[i];
    }
    printf("Sync sum = %f\n", sync_sum);

    float result = *h_sum_array; // first cell
    printf("h_sum_array[0] = %f\n", result);

    // print 6 first cell of h_sum_array
    for (int i = 0; i < 6; i++) {
        printf("%f ", h_sum_array[i]);
    }

    // free
    free(h_sum_array);
    cudaFree(d_sum_array);

    return result;
    //return sync_sum;
}

int main (int argc, char** argv)
{
    // declare variables
    static long num_steps = 100000000;
    int nb_blocks = 1;
    int threadsPerBlock = 64;
    float step;
    
    // Read command line arguments.
    for (int i = 0; i < argc; i++) {
        if ( 
            (strcmp(argv[i], "-N") == 0) || 
            (strcmp(argv[i], "-num_steps") == 0)
        ) {
            num_steps = atol(argv[ ++i ]);
            printf( "  User num_steps is %ld\n", num_steps );
        } else if ( 
            (strcmp(argv[i], "-B") == 0) || 
            (strcmp(argv[i], "-nb_blocks") == 0) 
        ) {
            nb_blocks = atol( argv[ ++i ] );
            printf( "  User nb_blocks is %d\n", nb_blocks );
        } else if ( 
            (strcmp(argv[i], "-T") == 0) || 
            (strcmp(argv[i], "-threadperblock") == 0) 
        ) {
            threadsPerBlock = atol( argv[ ++i ] );
            printf( "  User threadsPerBlock is %d\n", threadsPerBlock );
        } else if ( 
            (strcmp(argv[i], "-h") == 0) || 
            (strcmp(argv[i], "-help") == 0 ) 
        ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
    }

    // check params
    if (nb_blocks < 0 || nb_blocks > 65535) {
        printf("Warning with nb_blocks = %d\n", nb_blocks);
    }
    if (threadsPerBlock < 0 || threadsPerBlock > 1024) {
        printf("Warning with threadsPerBlock = %d\n", threadsPerBlock);
    }
    
    step = 1.0/(double) num_steps;

    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );

    // computation of PI below
    float sum = computePi(num_steps, step, nb_blocks, threadsPerBlock);

    float pi = step * sum;

    gettimeofday( &end, NULL );

    // Calculate time.
    float time = 1.0 * ( end.tv_sec - begin.tv_sec ) 
        + 1.0e-6 * ( end.tv_usec - begin.tv_usec );
    
    // output to file
    string result_str = 
        string("critical") + "," 
        + to_string(nb_blocks) + ","
        + to_string(num_steps) + ","
        + to_string(time);
    ofstream myfile("stats.csv", ios::app);
    if (myfile.is_open())
    {
        myfile << result_str << endl;
        myfile.close();
    }
    else cerr<<"Unable to open file";
    
    printf(
        "\n pi with %ld steps is %lf in %lf seconds\n ",
        num_steps, pi, time
    );

    return 0;
}
