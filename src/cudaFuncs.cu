
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
///////   This file contains cuda functions Implementation   /////////////////
//////////////////////////////////////////////////////////////////////////////
#include "cudaFuncs.h"
#include <cusolverDn.h>
#include <curand.h>
#include "math.h"

//This function does the cholesky decomposition
/* 
  input parameters: inMat: input matrix
                    dim: dimension of matrix
                    uplo: matrix fill type
*/
void chol(double* inMat, int dim, cublasFillMode_t uplo ){×
   //variables for cuSolver cholesky 
   cusolverDnHandle_t csrHandle = NULL;
   cusolverStatus_t status;[MaÔ[MaÔ[MaÔ[MaÔ
   bool TESTFLAG = false;
   //variables for workspace
   int workSize = 0;[MaÓ[MaÓ[MaÓ
   double* workPtr;

   int* devInfo; //used for error checking

   //allocate shared memory
   cudaMallocManaged( &devInfo, sizeof(int) );
   //create handle for library
   status = cusolverDnCreate( &csrHandle );
   //get buffer size[MaÓ[MaÓ
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
           printf(" %f", inM[MaÓ[MaÓ[MaÓ[MaÓat[ i*dim + j ]);
        }
        printf("\n");
     }[MaÓ[MaÓ[MaÓ
   }   

   //This step calls the cholesky function from cuSolver
     /* Function parameters: 
     cusolverDnHandle_t: handle to cuSolver library
     cublasFillMode_t: Indicates of matrix A lower or upper part stored
     int: dimension of matrix A
     float*: pointer to input matrix
     int: leading [MaÓ[MaÓdimension of 2D array used to store matrix
     float*:workspace pointer
     int: size of workspace
     int*: return for erro[MaÓr checking
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

//square matrix mult
//C = alpha*op(A)op(B) + beta*C
void matMult( double* matA, double* matB, double* outMat, int dim ){
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

  status = cublasDgemm( myHandle, transa, transb, dim, dim, dim, &one, matB,
                                            dim, matA, dim, &zero, outMat, dim );

  assert( status == CUBLAS_STATUS_SUCCESS );

  //destroy cublas instance
  cublasDestroy( myHandle );
}

//inverse CDF function, calls device function normcdfinv from CUDA math API
__global__ void invCDF( double* inMat, int n ){
  //initialzie variables for block id and thread id
    int bidx, tid;  
    bidx = blockIdx.x;
    tid = threadIdx.x;

    //grid stride loop
    for( int i = blockIdx.x * blockDim.x + threadIdx.x;  i < n; i+= blockDim.x * gridDim.x ){
       inMat[i] = normcdfinv( inMat[i] ); 
          
    }




}
