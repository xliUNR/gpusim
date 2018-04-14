///////////////////////////////////////////////////////////////////////////////
///////////////   This is the GPU version of NORTA   //////////////////////////
///////////////////// Written by Eric Li //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////



/////////////////////////////  Includes  //////////////////////////////////////
#include <cstdlib>
#include <fstream>
#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include <cusolverDn.h>
#include <assert.h>
#include <cuda.h>
//#include <curand.h>
#include "cudaFuncs.h"

using namespace std;






///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////    Main   ///////////////////////////////////
int main( int argc, char const *argv[])
{
 //initialize variables
   //initiliaze arrays for holding input data
   double* r20Arr; 
   //double* r200Arr;
   int r20n = 20;

   int r20Size; 
   //int r200Size;
   ifstream srcFile;
   float a;
   double A0[3*3] = { 1.0, 2.0, 3.0, 2.0, 5.0, 5.0, 3.0, 5.0, 12.0 };
   double* dA0;
   double* sim_data;

   //r200n = 200;
   r20Size = r20n*r20n;
   //r200Size = r200n*r200n;
   //cuSolver 
   //cuSolverStatus_t solverStatus;
   
   //cudaStream_t stream = NULL;   
   
   //set stream
   //cusolverDnSetStream(csrHandle, stream);

   //print cusolver version
   int major=-1,minor=-1,patch=-1;
    cusolverGetProperty(MAJOR_VERSION, &major);
    cusolverGetProperty(MINOR_VERSION, &minor);
    cusolverGetProperty(PATCH_LEVEL, &patch);
    printf("CUSOLVER Version (Major,Minor,PatchLevel): %d.%d.%d\n", major,minor,patch);

   //allocated unified memory for storage of input covar matrix. 
   cudaMallocManaged(&r20Arr, r20Size*sizeof(double));
   
   //allocate device memory for simple testing
   cudaMallocManaged( &dA0, 9*sizeof(double) );
   
   //copy explicitly defined matrix into device
   cudaMemcpy( dA0, A0, 9*sizeof(double), cudaMemcpyHostToDevice );
   //cudaMallocManaged(&r200Arr, r200Size*sizeof(float));
     
   //Timing for file read
   cudaEvent_t read readStart, readEnd;
   cudaEventCreate( &readStart, &readEnd );
   cudaEventRecord( readStart, 0); 

   //Section for reading in arrays from file
   srcFile.open("../test_corr_matrix_d=20.txt", fstream::in);
   if(srcFile)
      {
        cout << endl << "SUCCESSFUL FILE OPEN";
	 for(int i = 0; i < r20Size; i++)
          {
            srcFile >> a;
	          //cout << a << "|";
            if( !(i % 20) ){ cout << endl;}

            r20Arr[i] = a;
          } 
          
      }
    else
      {
        cout << std::endl << "ERROR OPENING FILE";
      }

 //close file
 srcFile.close();

 //stop timing
 cudaEventRecord( readEnd, 0 );
 cudaEventSynchronize( readEnd );
 float readTime;
 cudaEventElapsedTime( &readTime
 //test input read by printing results
  printf("\n INITIAL MATRIX\n");
 
  for(int i = 0; i < 3; i++ ){
    for(int j = 0; j <3; j++ )
      {
        printf(" %f", dA0[i*3+j]);
      } 
      printf("\n");
   } 
//cholesky decomp with floats (specified by S)
/*  //initialize variables
  cusolverDnHandle_t csrHandle = NULL;
  cublasFillMode_t uplo= CUBLAS_FILL_MODE_UPPER;
  cusolverStatus_t status;
  int r20workSize = 0;
  double* r20work;
  int* devInfo; //used for error checking
  
  cudaMallocManaged(&devInfo, sizeof(int));
  //double* r200work;
  //create cusolver handle 
  status = cusolverDnCreate(&csrHandle);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  //This is the Cholesky decomp step 
  //First calculate size of workspace
  */
  //float r200work;
  /*status = cusolverDnDpotrf_bufferSize(csrHandle, 
                                uplo, r20n, r20Arr, r20n, &r20workSize);
  status = cusolverDnDpotrf_bufferSize(csrHandle, uplo, 3, dA0, 3, &r20workSize);
  assert(CUSOLVER_STATUS_SUCCESS == status );

  //cusolverDnSpotrf_bufferSize(csrHandle, 
  //                              uplo, r200n, r200Arr, r200n, r200workSize);
 
  //Allocate memory for workspace
  cudaMallocManaged( &r20work, r20workSize*sizeof(double) );
  //cudaMallocManaged(&r200work, r200workSize*sizeof(float));
  
  //This step calls the cholesky function from cuSolver
     Function parameters: 
     cusolverDnHandle_t: handle to cuSolver library
     cublasFillMode_t: Indicates of matrix A lower or upper part stored
     int: dimension of matrix A
     float*: pointer to input matrix
     int: leading dimension of 2D array used to store matrix
     float*:workspace pointer
     int: size of workspace
     int*: return for error checking


  
  cusolverDnDpotrf(csrHandle, uplo, 3, dA0, 3, r20work, r20workSize, devInfo);
  cusolverDnDpotrf(csrHandle, uplo, r20n, r20Arr, r20n, 
                                      r20work, r20workSize, devInfo); 
  */
 
  //printf("Dev Info: %d", *devInfo);

//call function to perform cholesky
chol( dA0, 3, CUBLAS_FILL_MODE_UPPER );   

   //fclose(fp);
  /* fp = fopen("test_corr_matrix_d=200.txt", "r"); 
   if(fp)
      {
        for(int i = 0; i < 200 * 200; i++)
           {
             fscanf(fp, "%f", &r200[i]);
	   }
      }*/

   //test input read by printing results
  printf("\n DECOMP RESULTS: \n");
  for(int i = 0; i < 3; i++ ){
    for(int j = 0; j <3; j++ )
      {
        printf(" %f", dA0[i*3+j]);
      } 
      printf("\n");
   }
//generate random variables matrix
size_t n = 10;
double * randMat;
cudaMallocManaged( &randMat, 10*sizeof(double) );
normGen( randMat, n, 0.0,1.0 );

//print results to screen
printf("\n RANDOM MATRIX: \n");
for(int i = 0; i < 3; i++ ){
  for(int j=0; j < 3; j++){
    printf(" %f", randMat[i*3+j]);
    }
  printf("\n");
 }  

    /*for(int i = 0; i < 200; i++ ){
      for(int j = 0; j <200; j++ )
        {
          printf("%f", r200[i*20+j]);
        } 
        printf("\n");
   }  
     */ 
///////// generate random variable //////////////////////////////
//curandGenerateNormalDouble()
   //free memory
   cudaFree(r20Arr);
   //cudaFree(r20work);
   //cudaFree(r200);
   cudaFree(dA0);
   cudaFree(randMat);   
}
