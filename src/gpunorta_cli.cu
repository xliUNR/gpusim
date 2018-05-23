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
//This function reads correlation matrix from file
bool readFromFile(const char*, double*, int );
//This function reads list of distributions from file
bool readDistFile( const char*, distStruct*, int );
//This function fills in upper part of matrix with zeros
void fillZeros( double* inMat, int );
//This function prints simulation matrix to file
bool printMatToFile(char* , double* , int , int );




/////////////////////////// Main Loop  ////////////////////////////////////////
int main(int argc, char const *argv[])
{
    char corrFileName[60];
    char distFileName[60];
    char outputFName[60];
    char userRes;
    int d, n;
    double* corrMatrix;
    double* simMatrix;
    int corrSize, simSize;
    distStruct dists;

    //control grid and block structure for inverse transform kernel call
    dim3 grid(512);
    dim3 block(128);


   //copy command line params for correlation and distribution files 
   strcpy( corrFileName, argv[1] );
   strcpy( distFileName, argv[2] );

   //cout << endl << "correlations file name: " << corrFileName;
   //cout << endl << "distributions file name: " << distFileName;

   d = atoi( argv[3] );
   n = atoi( argv[4] );
   //cout << endl << "n: " << n;
   //cout << endl << "d: " << d;
         
   //set sizes for correlation matrix and simulation matrix
   corrSize = d*d;
   simSize = n*d;

  //meat of program goes here
    //first allocate memory for arrays and struct
  cudaMallocManaged( &corrMatrix, d*d*sizeof(double) );
  cudaMallocManaged( &simMatrix, n*d*sizeof(double) );
  cudaMallocManaged( &( dists.distKey ), d*sizeof(int) );
  cudaMallocManaged( &( dists.params ), d*sizeof(float*) );

   //cuda timing
  cudaEvent_t startTime, endTime;
  cudaEventCreate( &startTime );
  cudaEventCreate( &endTime );
  cudaEventRecord( startTime, 0 );
     

  //Read in correlation matrix from file
  if( !readFromFile( corrFileName, corrMatrix, corrSize ) ){
    cout << endl << "Error opening correlation matrix file! ";
  }

  //read in distributions from file
  if( !readDistFile( distFileName, &dists, d ) ){
    cout << endl << "Error opening Distribution file!";
  }

  //Cholesky decomp of correlation matrix
  chol( corrMatrix, d, CUBLAS_FILL_MODE_UPPER );
  //fill array with zeros
  fillZeros( corrMatrix, d );
  cout << endl << "Cholesky decomposition complete!";

  //generate i.i.d random variable matrix
  int time1 = time(NULL);
  normGen( simMatrix, simSize, 0.0, 1.0, time1 );
  cudaDeviceSynchronize();
  cout << endl << "Normal generation complete!";
    
  //multiply cholesky decomp w/ random var matrix to get correlated
  //random var
  matMult( simMatrix, corrMatrix, simMatrix, d, n, d );
  cudaDeviceSynchronize();
  cout << endl << "Matrix multiplication complete!";
    
  //perform inverse probability transformation
  invTransform<<<grid,block>>>( simMatrix, dists.distKey, dists.params, d, n );
  cout << endl << "Inverse transformation complete!";
    
  //End Timing
  cudaEventRecord( endTime );
  cudaEventSynchronize( endTime );
  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, startTime, endTime );
    
  //print timing results
  cout << endl << "The Program took: " << elapsedTime << " milliseconds to run.";   

  cout << endl << "Please enter in name of file to save";
  cout << " simulation matrix to: ";
  cin >> outputFName;

  if( printMatToFile( outputFName, simMatrix, n, d ) == true ){
    cout << endl << "Printing matrix to file success!";
  }
  else{
    cout << endl << "Error printing sim matrix to file";
  }


  //free memory
  cudaFree( corrMatrix );
  cudaFree( simMatrix );
  cudaFree( dists.distKey );
  for( int i = 0; i < d; i++ ){
     cudaFree( dists.params[i] ); 
  } 

  cudaFree( dists.params );


  return 0;
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
         //cout << "NAME OF DIST: ";
         //cout << endl << distName << endl;
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

        /* else if( strcmp( "chi-squared", distName ) == 0 ){
            dists->distKey[i] = 3;
            cudaMallocManaged( &( dists->params[i] ), 1*sizeof(float) );
            source >> dists->params[i][0];
         }*/

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

        /*else if( strcmp( "gamma", distName ) == 0 ){
            dists->distKey[i] = 6;
            cudaMallocManaged( &( dists->params[i] ), 2*sizeof(float) );
            source >> dists->params[i][0];
            source >> dists->params[i][1];
         } */

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


void fillZeros( double* inMat, int dim ){
  for(int i = 0; i < dim; i++){
    for(int j = 0; j < dim; j++){
      if( i < j ){
        inMat[ i*dim + j ] = 0;
      }
    }
  }
}

bool printMatToFile(char* outFileName, double* inMat, int rows, int cols ){
  ofstream fout;
  fout.open( outFileName, ofstream::out );
  //loop over all values of matrix and write them to file
  if( fout ){
    for(int i = 0; i < rows; i++ ){
      for(int j = 0; j < cols; j++ ){
        fout << inMat[ i*cols + j ] << ' ';
      }
      fout << endl;
    }
    fout.close();
    return true;
  }  
  else{
    return false;
  }
}