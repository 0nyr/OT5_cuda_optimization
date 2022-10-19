test.cpp
#include <cstdio>
#include <omp.h>
int main(int argc, char** argv) {
 int nb_core = 1;
 #pragma omp parallel
 {
       if (omp_get_thread_num() == 0) nb_core = omp_get_num_threads();
 }
 printf("%d\n",nb_core);
}

Then you compile : g++ -o test test.cpp

And you can test: 

OMP_NUM_THREADS=1 ./test

OMP_NUM_THREADS=4 ./test