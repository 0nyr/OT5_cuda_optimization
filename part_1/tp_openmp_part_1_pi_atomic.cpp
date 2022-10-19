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

static long num_steps = 100000000;
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
      
    int i;
    double x, pi, sum = 0.0;
    
    step = 1.0/(double) num_steps;

    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );

    // computation of PI below
    // read only values don't mind
    #pragma omp parallel for private(x) shared(sum)
    for (i=1; i<= num_steps; i++) {
        x = (i-0.5)*step;
        #pragma omp atomic
        sum = sum + 4.0/(1.0+x*x);
        // x = (i-0.5)*step;
        // x *= x;
        // x += 1.0;
        // x = 4.0/x;
        // #pragma omp atomic
        // sum = sum + x;
    }
    pi = step * sum;

    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) 
        + 1.0e-6 * ( end.tv_usec - begin.tv_usec );
    
    // output to file
    string result_str = 
        string("atomic") + "," 
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
