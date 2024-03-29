#ifndef CUDAFUNCS_H_
#define CUDAFUNCS_H_

#include <cuda.h>
#include <cstdlib>
#include <cstdio>
#include <assert.h>
#include <cusolverDn.h>
#include <curand.h>

void chol(double* , int , cublasFillMode_t );

void normGen( double*, size_t, double, double, int );

void matMult( double*, double*, double*, int, int, int );

__global__ void testFunc( double*, int );

__global__ void invTransform( double* , int* , float** , int , int );

__device__ double invTransformHelper( double, int, float* );

#endif
