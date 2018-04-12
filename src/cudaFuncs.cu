
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
///////   This file contains cuda functions Implementation   /////////////////
//////////////////////////////////////////////////////////////////////////////
#include "cudaFuncs.h"
#include <cusolverDn.h>
#include <curand.h>

void chol(double* inMat, int dim, cublasFillMode_t uplo ){
   //variables for cuSolver cholesky 
   cusolverDnHandle_t csrHandle = NULL;
   cusolverStatus_t status;
   
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
   printf("\n Matrix before decomp: \n");
   for(int i = 0; i < dim; i++ ){
      for(int j = 0; j < dim; j++ ){
         printf(" %f", inMat[ i*dim + j ]);
      }
      printf("\n");
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
  
  //print final results for checking
  printf("\n Matrix after decomp: \n");
   for(int i = 0; i < dim; i++ ){
      for(int j = 0; j < dim; j++ ){
         printf(" %f", inMat[ i*dim + j ]);
      }
      printf("\n");
   }
}

//This function generates pseudo random standard normal distribution
void normGen( double* outputPtr, int n ){
   //declare variables
   curandGenerator_t randHandle;
   curandStatus_t status;
   //First must create generator and set options
   status = curandCreateGenerator( &randHandle, CURAND_RNG_PSEUDO_XORWOW );
   assert( status == CURAND_STATUS_SUCCESS );
   //This step calls the random number generator function from cuRand
   /* Function paramters:
      curandGenerator_t : handle to generator
      float * outputPtr : pointer to array storing numbers
      size_t num        : 
      float mean        : Given mean
      float stddev      : Given standard deviation
   */
   status = curandGenerateNormalDouble( randHandle, outputPtr, n, 0, 0 );
   assert( status == CURAND_STATUS_SUCCESS );

   //print results for testing purposes
   printf("\n Psuedo random standard normal matrix: \n");
   for(int i = 0; i < n; i++ ){
      for(int j = 0; j < n; j++ ){
         printf(" %f", outputPtr[ i*n + j ]);
      }
      printf("\n");
   }
}
