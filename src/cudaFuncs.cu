//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
///////   This file contains cuda functions Implementation   /////////////////
//////////////////////////////////////////////////////////////////////////////
#include "cudaFuncs.h"
#include <cusolverDn.h>
#include 

void chol(int* inMat, int dim, cublasFillMode_t uplo ){
   //variables for cuSolver cholesky 
   cusolverDnHandle_t csrHandle = NULL;
   cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
   cusolverStatus_t status;
   
   //variables for workspace
   int workSize = 0;
   double* workPtr;

   int* devInfo //used for error checking

   //allocate shared memory
   cudaMallocManaged( &devInfo, sizeof(int) );
   //create handle for library
   status = cusolverDnCreate( &csrHandle );
   //get buffer size
   status = cussolverDnDpotrf_bufferSize(csrHandle, uplo, dim, 
                                                   inMat, dim, &workSize );
   assert( status == CUSOLVER_STATUS_SUCCESS );
   //allocate memory for workspace
   cudaMallocManaged( &workPtr, workSize * sizeof(double) );
   //print starting matrix for error checking
   printf("\n Matrix before decomp: ");
   for(int i = 0; i < dim; i++ ){
      for(int j = 0; j < dim, j++ ){
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
  printf("\n Matrix after decomp: ");
   for(int i = 0; i < dim; i++ ){
      for(int j = 0; j < dim, j++ ){
         printf(" %f", inMat[ i*dim + j ]);
      }
      printf("\n");
   }
}

//This function generates normal distributed
void normGen()