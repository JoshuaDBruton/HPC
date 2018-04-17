/**
histogram.c

Calculates the histogram of a random 2D matrix with integer
values in the range [0..255].

Compilation:
============
g++ histogram.c -fopenmp -o histogram

Execution:
==========
./histogram <n>

**/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h> 

#define BINS 256

void sHistogram(int **x, long *count, int N);
void pHistogram(int **x, long *count, int N);
int vectorEquil(long *vecA, long *vecB, int n);

int main(int argc,char **argv) {
	int n, i, j;
	int **x;
	long *scount, *pcount;
    float t, sT, pT, sUp;
    
	if(argc==2){
		n=atoi(argv[1]);
	} 
	else{
		printf("Enter n, the size of the nxn matrix.\n");
		exit(0);
	} 
	x = (int**)malloc(n*sizeof(int*));
    for(i = 0; i < n; i++) {
        x[i] = (int*)malloc(n*sizeof(int));
    }
	scount = (long*)malloc(BINS*sizeof(long));
	pcount = (long*)malloc(BINS*sizeof(long));
    
	srand(time(NULL)) ;
	//initialize
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            x[i][j] = rand()%BINS;
        }
    }
    
	//serial
	t=omp_get_wtime();
    sHistogram(x, scount, n);
	sT=omp_get_wtime()-t;
    
	//parallel 
	t=omp_get_wtime();
    pHistogram(x, pcount, n);
	pT=omp_get_wtime()-t;
    
	//verify the results from the serial and parallel versions 
	if(vectorEquil(scount,pcount,BINS)) {
		printf("Verifiaction failed!\n");
		exit(0);
    } else {
		printf("Verification passed!\n");
    }
	sUp=sT/pT;
	printf("\n---==== Problem size = %d x %d ====---\n", n, n);
    printf("Serial execution took %f seconds.\n", sT);
    printf("Parallel execution took %f seconds.\n", pT);
    printf("Speed up of %f.\n", sUp);
    
    for(i = 0; i < n; i++) free(x[i]);
	free(x);
	free(scount);
    free(pcount);
	return 0;
}

void sHistogram(int **x, long *count, int N) {
	int i, j; 
    // initialise histogram
    for(i = 0; i < BINS; i++) {
		count[i] = 0;
	}
    // count occurences
	for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            count[x[i][j]]++;
        }
	}
}


void pHistogram(int **x, long *count, int N) {
    // TODO: use OpenMP to parallelise this function
    
    int i, j; 
    // initialise histogram
    for(i = 0; i < BINS; i++) {
		count[i] = 0;
	}
    // count occurences
	#pragma omp parallel firstprivate(i,j) num_threads(8)	
	{
		long* local_sum = (long*)malloc(N*BINS);
		for (int i = 0;i<BINS;i++) local_sum[i]=0;
		#pragma omp for schedule(dynamic)
		for(i = 0; i < N; i++) {
		    for(j = 0; j < N; j++) {
		        local_sum[x[i][j]]++;
		    }
		}
		for (int i = 0; i<BINS; i++) {
			#pragma omp atomic
			count[i] = count[i] + local_sum[i];
		}	
	}
}

int vectorEquil(long *vecA, long *vecB, int n) {
	int bad=0;
	for(int i=0; i<n; i++){
		if(vecA[i] != vecB[i])
			bad++;
	}
	return bad;
}

