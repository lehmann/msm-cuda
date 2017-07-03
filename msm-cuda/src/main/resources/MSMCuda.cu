#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <sys/time.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__device__ double euclidean(double *p1, double *p2);

__global__ void msm(double* trajA, int lengthA, double* trajB, int lengthB, double* aScore, double* bScore, double* semanticsDescriptors);

int main() {
	int N = 10;
	double* trajA = (double*)malloc( N*N*sizeof(double));
	double* trajB = (double*)malloc( N*N*sizeof(double));
	double* semanticsDescriptors = (double*)malloc( 2*2*sizeof(double));
	double* aScore = (double*)malloc( N*sizeof(double));
	double* bScore = (double*)malloc( N*sizeof(double));
	struct timeval tv;
	gettimeofday(&tv, NULL);
	
	double time_in_mill = (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000 ; 
	for(int i = N - 1; i > -1; i--) {
		trajA[i * N] = i;
		trajA[i * N + 1] = i;
		trajA[i * N + 2] = time_in_mill - i;
		trajA[i * N + 3] = time_in_mill - (i-1);
		
		trajB[i * N] = i;
		trajB[i * N + 1] = i;
		trajB[i * N + 2] = time_in_mill - i;
		trajB[i * N + 3] = time_in_mill - (i-1);
	}
	//GEO
	semanticsDescriptors[0] = 0.0;
	semanticsDescriptors[1] = 0.5;
	//TIME
	semanticsDescriptors[2] = 0.0;
	semanticsDescriptors[3] = 0.5;
	
	double *d_trajA,*d_trajB, *d_aScore, *d_bScore, *d_semanticsDescriptors;
	cudaMalloc( (void**) &d_trajA, N*N*sizeof(double) );
	cudaMalloc( (void**) &d_trajB, N*N*sizeof(double) ); 

	cudaMalloc( (void**) &d_semanticsDescriptors, 2*2*sizeof(double) );
	cudaMalloc( (void**) &d_aScore, N*sizeof(double) );
	cudaMalloc( (void**) &d_bScore, N*N*sizeof(double) );
	cudaMemcpy( (void*) d_trajA, (void*) trajA, N*N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy( (void*) d_trajB, (void*) trajB, N*N*sizeof(double), cudaMemcpyHostToDevice); 
	cudaMemcpy( (void*) d_semanticsDescriptors, (void*) semanticsDescriptors, 2*2*sizeof(double), cudaMemcpyHostToDevice); 
	
	int THREADS = 128;
	int BLOCOS = (N/THREADS) + 1;
	
    struct timeval begin, end;
    gettimeofday(&begin, NULL);
    msm<<<BLOCOS, THREADS>>>( d_trajA, N, d_trajB, N, d_aScore, d_bScore, d_semanticsDescriptors );
    gettimeofday(&end, NULL);
    
	cudaMemcpy( (void*) aScore, (void*) d_aScore, N*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy( (void*) bScore, (void*) d_bScore, N*N*sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(d_trajA); 
	cudaFree(d_trajB); 
	cudaFree(d_aScore);
	cudaFree(d_bScore);
	cudaFree(d_semanticsDescriptors); 
	 
	double parityAB = 0.0;
	for (int i = 0; i < N; i++) {
		parityAB += aScore[i];
	}

	double parityBA = 0.0;
	for (int i = 0; i < N; i++) {
		double maxScore = 0.0;
		for (int j = 0; j < N; j++) {
			maxScore = MAX(maxScore, bScore[i * N + j]);
		}
		parityBA += maxScore;
	}
	//printf("parityAB=%.2f, parityBA=%.2f\n", parityAB, parityBA );
	double similarity = (parityAB + parityBA) / (N + N);
	
	printf("Similaridade das trajetórias: %.2f\n", similarity);
	
	return 0;
}

//extern "C"
__global__ void msm(double* trajA, int lengthA, double* trajB, int lengthB, double* aScore, double* bScore, double* semanticsDescriptors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=lengthA) {
    	return;
    }
	double latGeoA = trajA[i * lengthA];
	double lonGeoA = trajA[i * lengthA + 1];
	double startTimeA = trajA[i * lengthA + 2];
	double endTimeA = trajA[i * lengthA + 3];
	
	double geoThreshold = semanticsDescriptors[0];
	double timeThreshold = semanticsDescriptors[2];

	double geoWeight = semanticsDescriptors[1];
	double timeWeight = semanticsDescriptors[3];

	double maxScore = 0.0;
	double maxGeoScore = 0.0;
	double maxTimeScore = 0.0;
	for (int j = 0; j < lengthB; j++) {
		double latGeoB = trajB[j * lengthB];
		double lonGeoB = trajB[j * lengthB + 1];
		double startTimeB = trajB[j * lengthB + 2];
		double endTimeB = trajB[j * lengthB + 3];
		double timeScore = 0.0;
		if(startTimeA < endTimeB && startTimeB < endTimeA ) {
		    double overlap = MIN(endTimeA, endTimeB) - MAX(startTimeA, startTimeB);
		    if(overlap > 0.0) {
    			double duration = MAX(endTimeA, endTimeB) - MIN(startTimeA, startTimeB);
    			double timeDistance = 1 - (overlap / duration);
    			timeScore = (timeDistance <= timeThreshold ? 1 : 0) * timeWeight;
		    }
		}
		double geoB[] = {latGeoB, lonGeoB};
		double geoA[] = {latGeoA, lonGeoA};
		double geoScore = (euclidean(geoB, geoA) <= geoThreshold ? 1 : 0) * geoWeight;
		double sumScore = timeScore + geoScore;
		if(sumScore > maxScore) {
		    maxScore = sumScore;
		    maxGeoScore = geoScore;
		    maxTimeScore = timeScore;
		}
	    	bScore[i * lengthA + j] = sumScore;
	}
	//printf("Thread %d, maxScore=%.2f, maxGeoScore=%.2f, maxTimeScore=%.2f\n", i, maxScore, maxGeoScore,maxTimeScore );
	aScore[i] = maxScore;
}

__device__ double euclidean(double *p1, double *p2)
{
	double distX = abs(p1[0] - p2[0]);
	double distXSquare = distX * distX;

	double distY = abs(p1[1] - p2[1]);
	double distYSquare = distY * distY;

	return sqrt(distXSquare + distYSquare);
}

