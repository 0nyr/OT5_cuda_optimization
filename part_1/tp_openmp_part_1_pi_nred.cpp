/*

This program will numerically compute the integral of

                  4/(1+x*x) 
				  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/
#include <omp.h>
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

static unsigned long num_steps = 100000000;
int nb_threads = 1;
double step;

int main (int argc, char** argv)
{
    
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
    
    // setup OMP theads
    omp_set_num_threads(nb_threads); // WARN: in lowercase !!!
      
    double x, pi = 0.0;
    step = 1.0/(double) num_steps;
    unsigned long nb_red = num_steps/nb_threads;
    unsigned long rest = num_steps%nb_threads;
    unsigned long nb_compute_per_red = num_steps/nb_red;

    if (rest != 0)
    {
        nb_red++;
    }
    
    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );

    double sum = 0.0;
    #pragma omp parallel shared(sum)
    #pragma omp parallel for reduction(+: sum)
    for (size_t i = 1; i <= nb_red; i++)
    {
        double sum_red = 0.0;
        // computation of PI below
        // must be shared with reduction (see p.47/79)
        #pragma omp parallel private(x) shared(sum_red)
        #pragma omp for reduction(+: sum_red)
        for (size_t j = i*nb_compute_per_red; j < (i+1)*nb_compute_per_red; j++)
        {
            if (j <= num_steps){
                x = (j-0.5)*step;
                sum_red = sum_red + 4.0/(1.0+x*x);
            }
        }
        sum += sum_red;
    }
    pi = step * sum;

    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) 
        + 1.0e-6 * ( end.tv_usec - begin.tv_usec );
    
    // output to file
    string result_str = 
        string("nred") + "," 
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
