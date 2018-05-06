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
#include <string.h>

#include "cudaFuncs.h"
#include "stats.hpp"

using namespace std;


///////////////////////////  struct declaration  //////////////////////////////
//struct of arrays of distribution and their parameters
struct distStruct
   {
       int* distKey;
       float** params; 
   };

//////////////////////////// Function prototypes  /////////////////////////////
bool readFromFile(const char*, double*, int );
bool readDistFile( const char*, distStruct*, int );



///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////    Main   ///////////////////////////////////
int main( int argc, char const *argv[])
{
 //initialize variables
   //initiliaze arrays for holding input data
   double* r20Arr;
   double* r20ArrNF; 
   double* r200Arr;
   double* r20501Arr;

   int r20n = 20;
   int r200n = 200;
   int r20501n = 20501;
   int n = 1093;
   int r20Size; 
   int r200Size;
   int r20501Size;
   int sim20501Size = n * r20501n;
   int d = 6;

   //ifstream srcFile;
   double A0[3*3] = { 1.0, 2.0, 3.0, 2.0, 5.0, 5.0, 3.0, 5.0, 12.0 };
   double AC[6] = {1.0, 2.0, 3.0, 5.0, 5.0, 12.0};
   double AR[6] = {1.0, 2.0, 5.0, 3.0, 5.0, 12.0};
   double testdata = 0.1;
   double testArr[6] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
   double *dtestArr;
   double* dA0;
   double* dAR;
   double* dAC;
   double* sim_r20;
   double* sim_r200;
   double* sim_r20501;
   //file names
   char r20file[60] = "../test_corr_matrix_d=20.txt";
   char r200file[60] = "../test_corr_matrix_d=200.txt";
   char r20501file[60] = "../test_corr_matrix_d=20501.txt";
   char distFile[60] = "../distributions.txt";

   r20Size = r20n*r20n;
   r200Size = r200n*r200n;
   r20501Size = r20501n * r20501n;
   //initialize distribution struct for inverse prob
   distStruct dists;
   //initialize array for distributions
   cudaMallocManaged( &(dists.distKey), 6*sizeof(int) );
   
   //initialize array of pointers for parameters.
   cudaMallocManaged( &(dists.params), 6*sizeof(float*) );
   dists.params = new float*[ 6 ];

   //read in distributions file
   if( readDistFile( distFile, &dists, 6) ){
    cout << endl << "READ DIST FILE SUCCESS!";
   }

   /*ifstream src;
   char buffer[20];
   src.open(distFile, ifstream::in );
   for(int i = 0; i = 5; i++){
      src >> buffer;
      cout << buffer << ' ';
   }*/



   //cuSolver 
   //cuSolverStatus_t solverStatus;
   
   //cudaStream_t stream = NULL;   
   
   //set stream
   //cusolverDnSetStream(csrHandle, stream);

   /*//print cusolver version
   int major=-1,minor=-1,patch=-1;
    cusolverGetProperty(MAJOR_VERSION, &major);
    cusolverGetProperty(MINOR_VERSION, &minor);
    cusolverGetProperty(PATCH_LEVEL, &patch);
    printf("\n CUSOLVER Version (Major,Minor,PatchLevel): %d.%d.%d\n", major,minor,patch);*/

   //allocated unified memory for storage of input covar matrix. 
   cudaMallocManaged(&r20Arr, r20Size*sizeof(double));
   //cudaMallocManaged(&r20ArrNF, r20Size*sizeof(double));
   cudaMallocManaged(&r200Arr, r200Size*sizeof(double));
   cudaMallocManaged(&r20501Arr, r20501Size*sizeof(double));

   cudaMallocManaged(&sim_r20, r20Size*sizeof(double));
   cudaMallocManaged(&sim_r200, r200Size*sizeof(double));
   cudaMallocManaged(&sim_r20501, sim20501Size*sizeof(double));

   //allocate device memory for simple testing
   cudaMallocManaged( &dA0, 9*sizeof(double) );
   cudaMallocManaged( &dAR, 6*sizeof(double) );
   cudaMallocManaged( &dAC, 6*sizeof(double) );
   cudaMallocManaged( &dtestArr, 6*sizeof(double) );

   //copy explicitly defined matrix into device
   cudaMemcpy( dA0, A0, 9*sizeof(double), cudaMemcpyHostToDevice );
   cudaMemcpy( dAR, AR, 6*sizeof(double), cudaMemcpyHostToDevice );
   cudaMemcpy( dAC, AC, 6*sizeof(double), cudaMemcpyHostToDevice );
   cudaMemcpy( dtestArr, testArr, 6*sizeof(double), cudaMemcpyHostToDevice );



   //cudaMallocManaged(&r200Arr, r200Size*sizeof(float));
     
 //start timing
 
  //Timing for file read r20
   cudaEvent_t readStart, readEnd;
   cudaEventCreate( &readStart );
   cudaEventCreate(  &readEnd );
   cudaEventRecord( readStart, 0); 

 //call function to read in from file
 if( readFromFile( r20501file, r20501Arr, r20501Size) ){
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
 cout << endl << "Reading in r20501: " << readTime << " ms." << endl;

  /*//Timing for file read r200
   cudaEventCreate( &readStart );
   cudaEventCreate(  &readEnd );
   cudaEventRecord( readStart, 0); 
*/
 /*//call function to read in from file
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
 */
 //print timing results
 //cout << endl << "Reading in r200: " << readTime << " ms." << endl;
 
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

  //printf("Dev Info: %d", *devInfo);
//Timing for cholesky r20
cudaEvent_t cholStart, cholEnd;
cudaEventCreate( &cholStart ); 
cudaEventCreate( &cholEnd );
cudaEventRecord( cholStart, 0 );

//call function to perform cholesky
chol( r20501Arr, r20501n, CUBLAS_FILL_MODE_UPPER );   
//synchronize threads
cudaDeviceSynchronize();
//chol(dAR, 3, CUBLAS_FILL_MODE_LOWER );
//chol(dAC, 3, CUBLAS_FILL_MODE_LOWER );
//End timing
cudaEventRecord( cholEnd, 0);
cudaEventSynchronize( cholEnd );
float cholTime;
cudaEventElapsedTime( &cholTime, cholStart, cholEnd );
cout << endl << "Cholesky r20501 Took: " << cholTime << " ms." << endl;


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
//size_t n = 10;
double * randMat;
int time1 = time(NULL);
cudaMallocManaged( &randMat, 10*sizeof(double) );

//timing
cudaEvent_t randStart, randEnd;
cudaEventCreate( &randStart ); 
cudaEventCreate( &randEnd );
cudaEventRecord( randStart, 0 );

normGen( sim_r20501, sim20501Size, 0.0,1.0, time1 );

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
double* M3;
//allocate memory for matrix testing
cudaMallocManaged( &M1, 6*sizeof(double) );
cudaMallocManaged( &M2, 3*sizeof(double) );
cudaMallocManaged( &M3, 2*sizeof(double) );

for(int i = 0; i < 6; i ++){
  M1[i] = i;
}
for(int i = 0; i < 3; i++ ){
  M2[i] = i;
}

//parameters are: cols of M2, rows of M1, row of M2
matMult(M1, M2, M3, 1, 2, 3);

/*//print results
cout << endl << "MATRIX MULT RESULTS" << endl;
for(int i = 0; i < 2; i++ ){
  cout << endl << M3[i];
}
cout << endl;*/

 //multiplication of cholesky w/ random matrix to get correlated random matrix
 cudaEvent_t multStart, multEnd;
 cudaEventCreate( &multStart );
 cudaEventCreate( &multEnd );
 cudaEventRecord( multStart, 0 );
  
 matMult( sim_r20501, r20501Arr, sim_r20501, r20501n, n, r20501n );
 cudaEventRecord( multEnd, 0);
 cudaEventSynchronize( multEnd );
 float multTime;
 cudaEventElapsedTime( &multTime, multStart, multEnd );
 cout << endl << "mult r20: " << multTime << " ms." << endl;
 
 /*//calling qnorm from stats lib works.
 cout << "TESTING FOR STATS LIBRARY" << endl;
 cout << "b4 Value = " << testdata;
 cout << "after value = " << stats::qnorm(testdata) << endl; 
*/

//testing to see if library can be called from kernel
/*cout << "TESTING KERNEL " << endl;
cout << "B4 ARRAY : ";
for(int i = 0; i < 6; i++ ){
  cout << dtestArr[i] << ' ';
}

cout << endl;
int ffff  = 1;
dim3 grid(1);
dim3 block(6);

cudaDeviceSynchronize();
testFunc<<<grid,block>>>( dtestArr, 6 );
cudaDeviceSynchronize();
cout << "AFTER ARRAY: ";
for(int i = 0; i < 6; i++){
  cout << dtestArr[i] << ' ';
}
cout << endl;
 */
   //free memory
   cudaFree(r20Arr);
   cudaFree(r200Arr);
   cudaFree(r20501Arr);
   
   cudaFree(dA0);
   cudaFree(randMat); 
   cudaFree(M1);
   cudaFree(M2);
   cudaFree(M3);
   cudaFree(sim_r20);
   cudaFree(sim_r200);
   cudaFree(sim_r20501);
   cudaFree(dists.distKey);
   
   for( int i = 0; i < d; i++ ){
    cudaFree( dists.params[i] );
   }

   cudaFree( dists.params );
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

bool readDistFile(const char* fileName, distStruct* dists, int numDists ){
   ifstream source;
   char distName[20];
   source.open( fileName, fstream::in );
   float numBuffer;
   if( source ){
      //loop over all distributions 
      for( int i = 0; i < numDists; i++ ){ 
         source >> distName;
         cout << "NAME OF DIST: ";
         cout << endl << distName << endl;
         //test for each distribution supported, 14 total, 
         //sets params accordingly
         if( strcmp( "beta", distName) == 0 ){
            dists->distKey[i] = 0;
            cudaMallocManaged( &( dists->params[i] ), 2*sizeof(float) );
            source >> dists->params[i][0];
            source >> dists->params[i][1];
         }

         /*else if( strcmp( "binomial", distName) == 0 ){
            dists->distKey[i] = 1;
            dists->params[i] = new float[2];
            source >> dists->params[i][0];
            source >> dists->params[i][1];
         }*/

         else if( strcmp( "cauchy", distName ) == 0 ){
            dists->distKey[i] = 2;
            cudaMallocManaged( &( dists->params[i] ), 2*sizeof(float) );
            source >> dists->params[i][0];
            source >> dists->params[i][1];
         }

         else if( strcmp( "chi-squared", distName ) == 0 ){
            dists->distKey[i] = 3;
            cudaMallocManaged( &( dists->params[i] ), 1*sizeof(float) );
            source >> dists->params[i][0];
         }

         else if( strcmp( "exponential", distName ) == 0 ){
            dists->distKey[i] = 4;
            cudaMallocManaged( &( dists->params[i] ), 1*sizeof(float) );
            source >> dists->params[i][0];
         }

        else if( strcmp( "f", distName ) == 0 ){
            dists->distKey[i] = 5;
            cudaMallocManaged( &( dists->params[i] ), 2*sizeof(float) );
            source >> dists->params[i][0];
            source >> dists->params[i][1];
         } 

        else if( strcmp( "gamma", distName ) == 0 ){
            dists->distKey[i] = 6;
            cudaMallocManaged( &( dists->params[i] ), 2*sizeof(float) );
            source >> dists->params[i][0];
            source >> dists->params[i][1];
         } 

        else if( strcmp( "normal", distName ) == 0 ){
            dists->distKey[i] = 7;
            cudaMallocManaged( &( dists->params[i] ), 2*sizeof(float) );
            source >> dists->params[i][0] >> dists->params[i][1];
            //source >> dists->params[i][1];
         } 

        else if( strcmp( "lognormal", distName ) == 0 ){
            dists->distKey[i] = 8;
            cudaMallocManaged( &( dists->params[i] ), 2*sizeof(float) );
            source >> dists->params[i][0];
            source >> dists->params[i][1];
         }

        else if( strcmp( "logistic", distName ) == 0 ){
            dists->distKey[i] = 9;
            cudaMallocManaged( &( dists->params[i] ), 2*sizeof(float) );
            source >> dists->params[i][0];
            source >> dists->params[i][1];
         }

        else if( strcmp( "poisson", distName ) == 0 ){
            dists->distKey[i] = 10;
            cudaMallocManaged( &( dists->params[i] ), 1*sizeof(float) );
            source >> dists->params[i][0];
         }

        else if( strcmp( "t", distName ) == 0 ){
            dists->distKey[i] = 11;
            cudaMallocManaged( &( dists->params[i] ), 1*sizeof(float) );
            source >> dists->params[i][0];
         }

        else if( strcmp( "uniform", distName ) == 0 ){
            dists->distKey[i] = 12;
            cudaMallocManaged( &( dists->params[i] ), 2*sizeof(float) );
            source >> dists->params[i][0];
            source >> dists->params[i][1];
         }

        else if( strcmp( "weibull", distName ) == 0 ){
            dists->distKey[i] = 13;
            cudaMallocManaged( &( dists->params[i] ), 2*sizeof(float) );
            source >> dists->params[i][0];
            source >> dists->params[i][1];
         } 

        else{
          cout << endl << "Error reading in paramters, check spelling.";
          return false;
        }               
      }
      source.close();
      return true;  
   }

   else{
      cout << endl << "Error opening distributions file";
      source.close();
      return false;
   }
}
