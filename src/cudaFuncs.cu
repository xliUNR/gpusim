
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
///////   This file contains cuda functions Implementation   /////////////////
//////////////////////////////////////////////////////////////////////////////
#include "cudaFuncs.h"
#include <cusolverDn.h>
#include <curand.h>
#include "math.h"
#include "stats.hpp"



//This function does the cholesky decomposition
/* 
  input parameters: inMat: input matrix
                    dim: dimension of matrix
                    uplo: matrix fill type
*/
void chol(double* inMat, int dim, cublasFillMode_t uplo ){
   //variables for cuSolver cholesky 
   cusolverDnHandle_t csrHandle = NULL;
   cusolverStatus_t status;
   bool TESTFLAG = false;
   //variables for workspace
   int workSize = 0;
   double* workPtr;

   int* devInfo; //used for error checking

   //allocate shared memory
   cudaMallocManaged( &devInfo, sizeof(int) );
   //create handle for library
   status = cusolverDnCreate( &csrHandle );
   //get buffer size
   status = cusolverDnDpotrf_bufferSize(csrHandle, uplo, dim, 
                                                   inMat, dim, &workSize );
   assert( status == CUSOLVER_STATUS_SUCCESS );
   //allocate memory for workspace
   cudaMallocManaged( &workPtr, workSize * sizeof(double) );
   
   //print starting matrix for error checking
   if(TESTFLAG){
     printf("\n Matrix before decomp: \n");
     for(int i = 0; i < dim; i++ ){
        for(int j = 0; j < dim; j++ ){
           printf(" %f", inMat[ i*dim + j ]);
        }
        printf("\n");
     }
   }   

   //This step calls the cholesky function from cuSolver
     /* Function parameters: 
     cusolverDnHandle_t: handle to cuSolver library
     cublasFillMode_t: Indicates of matrix A lower or upper part stored
     int: dimension of matrix A
     float*: pointer to input matrix
     int: leading dimension of 2D array used to store matrix
     float*:workspace pointer
     int: size of workspace
     int*: return for error checking
     */ 
  status = cusolverDnDpotrf(csrHandle, uplo, dim, inMat, dim, 
                                          workPtr, workSize, devInfo);
  
   assert( status == CUSOLVER_STATUS_SUCCESS );  

  printf("\n Dev Info for cholesky: %d", *devInfo);
  
  if(TESTFLAG){
    //print final results for checking
    printf("\n Matrix after decomp: \n");
     for(int i = 0; i < dim; i++ ){
        for(int j = 0; j < dim; j++ ){
           printf(" %f", inMat[ i*dim + j ]);
        }
        printf("\n");
     }
  } 

  //destroy cusolver handle
  status = cusolverDnDestroy( csrHandle );
  assert( status == CUSOLVER_STATUS_SUCCESS );
}


//This function generates pseudo random standard normal distribution
void normGen( double* outputPtr, size_t n, double mean, double stddev, int seed ){
   //declare variables
   curandGenerator_t randHandle;
   curandStatus_t status;
   //First must create generator and set options
   status = curandCreateGenerator( &randHandle, CURAND_RNG_PSEUDO_DEFAULT );
   assert( status == CURAND_STATUS_SUCCESS && "create generator");
   //This step calls the random number generator function from cuRand
   /* Function paramters:
      curandGenerator_t : handle to generator
      float * outputPtr : pointer to array storing numbers
      size_t num        : 
      double mean       : Given mean
      double stddev     : Given standard deviation
   */
   status = curandSetPseudoRandomGeneratorSeed( randHandle, seed );
   assert( status == CURAND_STATUS_SUCCESS && "seeder");

   status = curandGenerateNormalDouble( randHandle, outputPtr, n, mean, stddev );
   assert( status == CURAND_STATUS_SUCCESS && "curand function call");

   //print results for testing purposes
   /*printf("\n Psuedo random standard normal matrix: \n");
   for(int i = 0; i < 3; i++ ){
      for(int j = 0; j < 3; j++ ){
         printf(" %f", outputPtr[ i*n + j ]);
      }
      printf("\n");
   }*/
   status = curandDestroyGenerator( randHandle );
   assert( status == CURAND_STATUS_SUCCESS && "destroyer" );
}

//matrix mult
//C = alpha*op(A)op(B) + beta*C
//m is cols of matrix B
//n is rows of matrix A
//k is rows of matrix B
void matMult( double* matA, double* matB, double* outMat, int m, int n, int k ){
  //declare variables
  cublasHandle_t myHandle;
  cublasStatus_t status;
  double zero = 0;
  double one = 1;
  //variables for if matrix is normal, transpose, or hermitian t.
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  //create library instance
  status = cublasCreate( &myHandle );
  assert( status == CUBLAS_STATUS_SUCCESS );

  status = cublasDgemm( myHandle, transa, transb, m, n, k, &one, matB,
                                            m, matA, k, &zero, outMat, m );

  assert( status == CUBLAS_STATUS_SUCCESS );

  //destroy cublas instance
  cublasDestroy( myHandle );
}


//function for testing calling stat library from kernel
__global__ void testFunc( double* inMat, int cols ){
  //int bidx, tid;
  double temp;
  printf(" \n cols %d", cols );
  //temp = stats::qnorm( inMat[ blockIdx.x * blockDim.x + threadIdx.x ] );
  //grid stride
  for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < cols; i+= blockDim.x*gridDim.x ){
    
    inMat[i] = stats::qnorm( inMat[i] );
    printf("Index: %d | %f ", i, inMat[i] );
  }
  
}

//function for inverse transform
__global__ void invTransform( double* simData, int* distArrPtr, 
                                              float** paramArr, int d, int n ){
  printf(" d ; %d \n", d);
  printf(" n ; %d \n", n);
  
  //Stride loop
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < d*n; 
                                        i+= blockDim.x * gridDim.x ){
    //first find cdf for normal dist
    simData[ i ] = normcdf( simData[ i ] );
    //Then do the inverse transform to specified marginal
    simData[ i ] =
    invTransformHelper( simData[ i ], distArrPtr[ (i/d) ], paramArr[ (i/d) ] );
    printf( "matrix elem: %d data: %f \n", blockIdx.x * blockDim.x + threadIdx.x, simData[i] );
  }
}


//helper function that calls stats package functions and returns calc'd value
__device__ double invTransformHelper( double val, int key, float* paramsArr ){
  double returnVal;
  //int nTrials = 7;
  switch( key ){
    case 0:
      returnVal = stats::qbeta( val, paramsArr[0], paramsArr[1] );
      printf("hey 0 worked \n");
      break; 

    case 1:
      //nTrials = paramsArr[0];
      //returnVal = stats::qbinom( val, nTrials, paramsArr[1] );
      break;

    case 2:
      returnVal = stats::qcauchy( val, paramsArr[0], paramsArr[1] );
      printf("hey 2 worked \n");
      break;  
    
    case 3:
      returnVal = stats::qchisq( val, paramsArr[0] );
      printf("hey 3 worked \n");
      break;

    case 4:
      returnVal = stats::qexp( val, paramsArr[0] );
      printf("hey 4 worked \n");
      break;
      
    case 5:
      returnVal = stats::qf( val, paramsArr[0], paramsArr[1] );
      printf("hey 5 worked \n");
      break;
      
    case 6:
      returnVal = stats::qgamma( val, paramsArr[0], paramsArr[1] );
      printf("hey 6 worked \n");
      break;
      
    case 7:
      returnVal = stats::qnorm( val, paramsArr[0], paramsArr[1] );
      printf("hey 7 worked \n");
      break;
      
    case 8:
      returnVal = stats::qlnorm( val, paramsArr[0], paramsArr[1] );
      printf("hey 8 worked \n");
      break;
      
    case 9:
      returnVal = stats::qlogis( val, paramsArr[0], paramsArr[1] );
      printf("hey 9 worked \n");
      break;
      
    case 10:
      returnVal = stats::qpois( val, paramsArr[0] );
      printf("hey 10 worked \n");
      break;
      
    case 11:
      returnVal = stats::qt( val, paramsArr[0] );
      printf("hey 11 worked \n");
      break;
      
    case 12:
      returnVal = stats::qunif( val, paramsArr[0], paramsArr[1] );
      printf("hey 12 worked \n");
      break; 

    case 13:
      returnVal = stats::qweibull( val, paramsArr[0], paramsArr[1] );
      printf("hey 13 worked \n");
      break;                      
  }

  return returnVal;
}
