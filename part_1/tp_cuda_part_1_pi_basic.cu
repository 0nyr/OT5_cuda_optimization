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

__global__ void computePiKernel(
    long num_steps, 
    float step,
    unsigned long nbComputePerBlock,
    float * d_sum
) { 
    long i = threadIdx.x + blockDim.x*blockIdx.x;

    float tmp_block_sum = 0.0;
    float x;
    for (
        int j = i*nbComputePerBlock; 
        j < (i + 1)*nbComputePerBlock; j++
    ) {
        if (j <= num_steps) {
            x = (j-0.5)*step;
            tmp_block_sum = tmp_block_sum + 4.0/(1.0+x*x);
        }
    }

    // write atomic to d_sum
    atomicAdd(d_sum, tmp_block_sum);
}

float computePi(
    long num_steps, 
    float step,
    int nb_threads
) {
    // memory allocations
    float * h_sum = (float *) malloc(sizeof(double)); // host (CPU)
    float * d_sum; // device (GPU)
    cudaError_t err = cudaMalloc((float **) &d_sum, sizeof(double));
    if (err != cudaSuccess) {
        printf(
            "%s in %s at line %d\n", cudaGetErrorString(err),
            __FILE__,
            __LINE__
        );
        exit(EXIT_FAILURE);
    }

    // prepare computing
    unsigned long nbComputePerBlock = num_steps / nb_threads;

    // do computation in device (GPU)
    computePiKernel<<<nb_threads, 1>>>(
        num_steps, step, nbComputePerBlock, d_sum
    );
    cudaDeviceSynchronize(); // kernel functions are async

    // get back result from device
    cudaMemcpy(h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    float result = *h_sum;

    // free
    free(h_sum);
    cudaFree(d_sum);

    return result;
}

int main (int argc, char** argv)
{
    // declare variables
    static long num_steps = 100000000;
    int nb_threads = 1;
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
            (strcmp(argv[i], "-T") == 0) || 
            (strcmp(argv[i], "-nb_threads") == 0) 
        ) {
            nb_threads = atol( argv[ ++i ] );
            printf( "  User nb_threads is %d\n", nb_threads );
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
    
    step = 1.0/(double) num_steps;

    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );

    // computation of PI below
    float sum = computePi(num_steps, step, nb_threads);

    float pi = step * sum;

    gettimeofday( &end, NULL );

    // Calculate time.
    float time = 1.0 * ( end.tv_sec - begin.tv_sec ) 
        + 1.0e-6 * ( end.tv_usec - begin.tv_usec );
    
    // output to file
    string result_str = 
        string("critical") + "," 
        + to_string(nb_threads) + ","
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
}
