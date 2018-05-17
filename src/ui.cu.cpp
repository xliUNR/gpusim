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
       int* distKeys;
       float** params; 
   };

//////////////////////////// Function prototypes  /////////////////////////////
//This function reads correlation matrix from file
bool readFromFile(const char*, double*, int );
//This function reads list of distributions from file
bool readDistFile( const char*, distStruct*, int );
//This function fills in lower part of matrix with zeros
void fillZeros( double* inMat, int dim );




/////////////////////////// Main Loop  ////////////////////////////////////////
int main(int argc, char const *argv[])
{
    bool progRepeat = true;
    bool askRepeat = true;
    bool fileRep = true;
    char corrFileName[60];
    char distFileName[60];
    char userRes[10];
    int d, n;
    double* corrMatrix;
    double* simMatrix;
    int corrSize, simSize;
    distStruct dists;
    //vary grid and block structure for inverse transform kernel call
    dim3 grid(512);
    dim3 block(128);

    //keep repeating if flag is true
    while(progRepeat == true ){
      //ask for input arguments if user didn't provide through command line  
      if(argc <= 1 ){
              //ask user for input
        cout << endl << "Enter in correlation matrix file name: ";
        cin >> corrFileName;

        cout << endl << "Enter in distributions file name: ";
        cin >> distFileName;

        cout << endl << "Enter in dimension of correlation matrix: ";
        cin >> d;

        cout << endl << "Enter in Number of simulation replicates: ";
        cin >> n;
        }
      else{
        corrFileName = argv[1];
        distFileName = argv[2];  
        d = argv[3];
        n = argv[4];
      }    
        //set sizes for correlation matrix and simulation matrix
        corrSize = d*d;
        simSize = n*d;

        //meat of program goes here
          //first allocate memory for arrays and struct
        cudaMallocManaged( &corrMatrix, d*d*sizeof(double) );
        cudaMallocManaged( &simMatrix, n*d*sizeof(double) );
        cudaMallocManaged( &( dists.distKey ), d*sizeof(int) );
        cudaMallocManaged( &( dists.params ), d*sizeof(float*) );

          //Read in correlation matrix from file
          while( !readFromFile( corrFileName, corrMatrix, n*d ) && fileRep ){
             cout << endl << "Error opening correlation matrix file! ";
             cout << endl << "Would you like to try again? (y/n)";
             cin >> userRes;

              if( userRes == 'y' | userRes == 'Y'){
                 cout << endl << "Please enter in filename: ";
                 cin >> corrFileName;  
              }
              else if( userRes == 'n' | userRes == 'N' ){
                 fileRep = false;
                 exit();
              }
              else{
                  cout << endl <<"Please enter in a valid response y or n.";
              }

          } 

          //cuda timing
          cudaEvent_t startTime, endTime;
          cudaEventCreate( &startTime );
          cudaEventCreate( &endTime );
          cudaEventRecord( startTime, 0 );
           
          //read in distributions from file
          readDistFile( distFileName, d );

          //Cholesky decomp of correlation matrix
          chol( corrMatrix, d, CUBLAS_FILL_MODE_UPPER );

          //generate i.i.d random variable matrix
          int time1 = time(NULL);
          normGen( simMatrix, simSize, 0.0, 1.0, time1 );
          cudaDeviceSynchronize();

          
          //multiply cholesky decomp w/ random var matrix to get correlated
          //random var
          matMult( simMatrix, corrMatrix, d, n, d );
          cudaDeviceSynchronize();

          
          //perform inverse probability transformation
          invTransform<<<grid,block>>>( simMatrix, dists.distKeys, dists.params, d, n );

          
          //End Timing
          cudaEventRecord( endTime );
          cudaEventSynchronize( endTime );
          float elapsedTime;
          cudaEventElapsedTime( &elapsedTime, startTime, endTime );
          
          //print timing results
          cout << endl << "The Program took: " << elapsedTime << "milliseconds to run.";   

        While( askRepeat ){
           cout << endl << "Would you like to run program again? (y/n)? ";
           cin >> userResponse;

           if( userRes == 'y' | userRes == "yes" 
                    | userRes == 'Y' | userRes == "Yes" | userRes == "YES"){
              askRepeat = false; 
           }

           else if( userRes == 'n' | userRes == 'N' | userRes == "no"
                    | userRes == "No" | userRes == "NO" ){
              askRepeat = false;
              progRepeat = false; 
           }
           else{
              cout << endl << "Invalid response, please answer y or n"; 
           }
        }
       //free memory
       cudaFree( corrMatrix );
       cudaFree( simMatrix );
       cudaFree( dists.distKey );
       for( int i = 0; i < d; i++ ){
          cudaFree( dists.params[i] ); 
       } 

       cudaFree( dists.params );

    }
    return 0;
}
