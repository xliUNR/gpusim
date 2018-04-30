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
#include <time.h>
//#include <curand.h>
#include "cudaFuncs.h"

using namespace std;


//////////////////////////// Function prototypes  /////////////////////////////
bool readFromFile(const char*, double*, int );

///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////    Main   ///////////////////////////////////
int main( int argc, char const *argv[])
{
 //initialize variables
   //initiliaze arrays for holding input data
   double* r20Arr;
   double* r20ArrNF; 
   double* r200Arr;
   double* r200ArrNF;

   int r20n = 20;
   int r200n = 200;
   int r20Size; 
   int r200Size;
   //ifstream srcFile;
   double A0[3*3] = { 1.0, 2.0, 3.0, 2.0, 5.0, 5.0, 3.0, 5.0, 12.0 };
   double AC[6] = {1.0, 2.0, 3.0, 5.0, 5.0, 12.0};
   double AR[6] = {1.0, 2.0, 5.0, 3.0, 5.0, 12.0};
   double* dA0;
   double* dAR;
   double* dAC;
   double* sim_r20;
   double* sim_r200;
   //file names
   char r20file[60] = "../test_corr_matrix_d=20.txt";
   char r200file[60] = "../test_corr_matrix_d=200.txt";

   r20Size = r20n*r20n;
   r200Size = r200n*r200n;
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
   //cudaMallocManaged(&r20ArrNF, r20Size*sizeof(double));
   cudaMallocManaged(&r200Arr, r200Size*sizeof(double));
   //cudaMallocManaged(&r200ArrNF, r200Size*sizeof(double));

   cudaMallocManaged(&sim_r20, r20Size*sizeof(double));
   cudaMallocManaged(&sim_r200, r200Size*sizeof(double));

   //allocate device memory for simple testing
   cudaMallocManaged( &dA0, 9*sizeof(double) );
   cudaMallocManaged( &dAR, 6*sizeof(double) );
   cudaMallocManaged( &dAC, 6*sizeof(double) );

   //copy explicitly defined matrix into device
   cudaMemcpy( dA0, A0, 9*sizeof(double), cudaMemcpyHostToDevice );
   cudaMemcpy( dAR, AR, 6*sizeof(double), cudaMemcpyHostToDevice );
   cudaMemcpy( dAC, AC, 6*sizeof(double), cudaMemcpyHostToDevice );

   //cudaMallocManaged(&r200Arr, r200Size*sizeof(float));
     

   /*//Section for reading in arrays from file
   srcFile.open("../test_corr_matrix_d=200.txt", fstream::in);
   if(srcFile)
      {
        cout << endl << "SUCCESSFUL FILE OPEN";
	 for(int i = 0; i < r200Size; i++)
          {
            srcFile >> r200ArrNF[i];
          } 
          
      }
    else
      {
        cout << std::endl << "ERROR OPENING FILE";
      }

 //close file
 srcFile.close();
*/
 //start timing
 
  //Timing for file read r20
   cudaEvent_t readStart, readEnd;
   cudaEventCreate( &readStart );
   cudaEventCreate(  &readEnd );
   cudaEventRecord( readStart, 0); 

 //call function to read in from file
 if( readFromFile( r20file, r20Arr, r20Size) ){
   cout << endl << "FILE OPEN SUCCESS!";
 }  
 else{
   cout << endl << "ERROR FILE OPENING";
 }
 /*
 //stop timing
 cudaEventRecord( readEnd, 0 );
 cudaEventSynchronize( readEnd );
 float readTime;
 cudaEventElapsedTime( &readTime, readStart, readEnd );
 //print timing results
 cout << endl << "Reading in r20: " << readTime << " ms." << endl;
*/
  /*//Timing for file read r200
   cudaEventCreate( &readStart );
   cudaEventCreate(  &readEnd );
   cudaEventRecord( readStart, 0); 
*/
 //call function to read in from file
 if( readFromFile( r200file, r200Arr, r200Size) ){
   cout << endl << "FILE OPEN SUCCESS!";
 }  
 else{
   cout << endl << "ERROR FILE OPENING";
 }
 
 //stop timing
 cudaEventRecord( readEnd, 0 );
 cudaEventSynchronize( readEnd );
 float readTime;
 cudaEventElapsedTime( &readTime, readStart, readEnd );
 
 //print timing results
 cout << endl << "Reading in r200: " << readTime << " ms." << endl;
 
/* //test input read by printing results
  printf("\n INITIAL MATRIX\n");
 
  for(int i = 0; i < 20; i++ ){
    for(int j = 0; j <20; j++ )
      {
        printf(" %f", r20Arr[i*3+j]);
      } 
      printf("\n");
   }
*/

//cholesky decomp with floats (specified by S)
/*  //initialize variables
  cusolverDnHandle_t csrHandle = NULL;
  cublasFillMode_t jplo= CUBLAS_FILL_MODE_UPPER;
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
//Timing for cholesky r20
cudaEvent_t cholStart, cholEnd;
cudaEventCreate( &cholStart ); 
cudaEventCreate( &cholEnd );
cudaEventRecord( cholStart, 0 );

//call function to perform cholesky
//chol( r200Arr, r200n, CUBLAS_FILL_MODE_UPPER );   
//synchronize threads
cudaDeviceSynchronize();
chol(dAR, 3, CUBLAS_FILL_MODE_LOWER );
cudaDeviceSynchronize();
chol(dAC, 3, CUBLAS_FILL_MODE_LOWER );
//End timing
cudaEventRecord( cholEnd, 0);
cudaEventSynchronize( cholEnd );
float cholTime;
cudaEventElapsedTime( &cholTime, cholStart, cholEnd );
cout << endl << "Cholesky r20 Took: " << cholTime << " ms." << endl;


//Timing for cholesky r200
//cudaEventCreate( &cholStart ); 
//cudaEventCreate( &cholEnd );
//cudaEventRecord( cholStart, 0 );

/*//call function to perform cholesky
chol( r200Arr, 200, CUBLAS_FILL_MODE_UPPER );   
//synchronize threads
cudaDeviceSynchronize();

//End timing
cudaEventRecord( cholEnd, 0);
cudaEventSynchronize( cholEnd );
float cholTime1;
cudaEventElapsedTime( &cholTime, cholStart, cholEnd );
cout << endl << "Cholesky r200 Took: " << cholTime1 << " ms." << endl;
*/
   //fclose(fp);
  /* fp = fopen("test_corr_matrix_d=200.txt", "r"); 
   if(fp)
      {
        for(int i = 0; i < 200 * 200; i++)
           {
             fscanf(fp, "%f", &r200[i]);
	   }
      }*/

   /*//test input read by printing results
  printf("\n DECOMP RESULTS: \n");
  for(int i = 0; i < 3; i++ ){
    for(int j = 0; j <3; j++ )
      {
        printf(" %f", dA0[i*3+j]);
      } 
      printf("\n");
   }*/
///////// generate random variable //////////////////////////////
size_t n = 10;
double * randMat;
int time1 = time(NULL);
cudaMallocManaged( &randMat, 10*sizeof(double) );

//timing
cudaEvent_t randStart, randEnd;
cudaEventCreate( &randStart ); 
cudaEventCreate( &randEnd );
cudaEventRecord( randStart, 0 );

normGen( sim_r200, r200Size, 0.0,1.0, time1 );

cudaEventRecord( randEnd, 0);
cudaEventSynchronize( randEnd );
float randTime;
cudaEventElapsedTime( &randTime, randStart, randEnd );
cout << endl << "RNG r200: " << randTime << " ms." << endl;

cout <<endl << "TIME SEED: " << time1;
/*//print results to screen
printf("\n RANDOM MATRIX: \n");
for(int i = 0; i < 200; i++ ){
  for(int j=0; j < 200; j++){
    printf(" %f", sim_r200[i*3+j]);
    }
  printf("\n");
 }  */

    /*for(int i = 0; i < 200; i++ ){
      for(int j = 0; j <200; j++ )
        {
          printf("%f", r200[i*20+j]);
        } 
        printf("\n");
   }  
     */ 
/////////////////////  matrix multiplication  /////////////////////////////////
/*cudaEvent_t multStart, multEnd;
cudaEventCreate( &multStart );
cudaEventCreate( &multEnd );
cudaEventRecord( multStart, 0 );*/
double* M1;
double* M2;

//allocate memory for matrix testing
cudaMallocManaged( &M1, 4*sizeof(double) );
cudaMallocManaged( &M2, 4*sizeof(double) );

//fill matrices with values
for(int i = 0; i < 4; i++ ){
  M1[i] = 2.0;
  M2[i] = 2.0;
}

matMult(M1, M2, M1, 2);

//print results
cout << "MATRIX MULT RESULTS" << endl;
for(int i = 0; i < 2; i++ ){
  for(int j = 0; j < 2; j++ ){
    cout << M1[i*2+j];
  }
  cout << endl;
}

 //multiplication of cholesky w/ random matrix to get correlated random matrix
 cudaEvent_t multStart, multEnd;
 cudaEventCreate( &multStart );
 cudaEventCreate( &multEnd );
 cudaEventRecord( multStart, 0 );
 
 matMult(r200Arr, sim_r200, sim_r200, r200n ); 

 cudaEventRecord( multEnd, 0);
 cudaEventSynchronize( multEnd );
 float multTime;
 cudaEventElapsedTime( &multTime, multStart, multEnd );
 cout << endl << "mult r20: " << multTime << " ms." << endl;

 
 
   //free memory
   cudaFree(r20Arr);
   cudaFree(r200Arr);
   //cudaFree(r20work);
   //cudaFree(r200);
   cudaFree(dA0);
   cudaFree(randMat); 
   cudaFree(M1);
   cudaFree(M2);
   cudaFree(sim_r20);
   cudaFree(sim_r200);
}

/////////////////// Function Implementation ///////////////////////////////////
bool readFromFile( const char* fileName, double* output, int size ){
   ifstream source;
   source.open( fileName, fstream::in );

   if( source ){
       for( int i = 0; i < size; i++ )
         {
            source >> output[i];
         }
       source.close();  
       return true;
     }

   else{
     source.close();   
     return false;
     }

}
