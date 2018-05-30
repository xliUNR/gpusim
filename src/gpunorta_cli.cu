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
#include <math.h>
#include <random>
#include "cudaFuncs.h"
#include "stats.hpp"
#include "book.h"

using namespace std;


/////////////////////// Macro Definition  /////////////////////////////////////
#define SEQUENTIAL 0
#define RUNINVERSE 1
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
//This function is the sequential version of the cholesky
void seqChol( double*, int );
//This function transposes input matrix
void matrixT(double* inMat, int , int );
//This function takes in a matrix and generates random numbers from normal dist
void seqNormGen( double*, int, int, int );
//This function does sequential matrix multiplication
void seqMatrixMult(double* , double* , double* , int , int , int );
//This function performs inverse transformation sequentially
void seqInvTransform( double* , int* , float** , int , int );
//This is the helper function for sequential inverse transform
double seqInvTransformHelper( double, int, float* );


/////////////////////////// Main Loop  ////////////////////////////////////////
int main(int argc, char const *argv[])
{
    char corrFileName[60];
    char distFileName[60];
    char outputFName[60];
    char userRes;
    int d, n, seed, runType, invTransFlag;
    double* corrMatrix;
    double* simMatrix;
    int corrSize, simSize;
    distStruct dists;

    //control grid and block structure for inverse transform kernel call
    //dim3 grid(512);
    //dim3 block(128);


   //copy command line params for correlation and distribution files 
   strcpy( corrFileName, argv[1] );
   strcpy( distFileName, argv[2] );

   //cout << endl << "correlations file name: " << corrFileName;
   //cout << endl << "distributions file name: " << distFileName;

   d = atoi( argv[3] );
   n = atoi( argv[4] );
   seed = atoi( argv[5] );
   //copy name of output file name for simulation matrix
   strcpy( outputFName, argv[6] );

   runType = atoi( argv[7] );
   invTransFlag = atoi( argv[8] );

         
   //set sizes for correlation matrix and simulation matrix
   corrSize = d*d;
   simSize = n*d;

  //meat of program goes here
    //first allocate memory for arrays and struct
  cudaMallocManaged( &corrMatrix, corrSize*sizeof(double) );
  cudaMallocManaged( &simMatrix, simSize*sizeof(double) );
  cudaMallocManaged( &( dists.distKey ), d*sizeof(int) );
  cudaMallocManaged( &( dists.params ), d*sizeof(float*) );


                    //Read in correlation matrix 
///////////////////////////////////////////////////////////////////////////////
   //start timing for correlation matrix read
  cudaEvent_t corrReadStart, corrReadEnd;
  cudaEventCreate( &corrReadStart );
  cudaEventCreate( &corrReadEnd );
  cudaEventRecord( corrReadStart, 0 );  

  //Read in correlation matrix from file
  if( !readFromFile( corrFileName, corrMatrix, corrSize ) ){
    cout << endl << "Error opening correlation matrix file! ";
  }

  //End Timing
  cudaEventRecord( corrReadEnd, 0 );
  cudaEventSynchronize( corrReadEnd );
  float corrReadTime;
  cudaEventElapsedTime( &corrReadTime, corrReadStart, corrReadEnd );
  cout << endl << "Reading correlation file timing: ";
  cout << corrReadTime << " ms." << endl;
///////////////////////////////////////////////////////////////////////////////

 
                    //Read in distribution file
///////////////////////////////////////////////////////////////////////////////
  //start timing for correlation matrix read
  cudaEvent_t distReadStart, distReadEnd;
  cudaEventCreate( &distReadStart );
  cudaEventCreate( &distReadEnd );
  cudaEventRecord( distReadStart, 0 );

  //read in distributions from file
  if( !readDistFile( distFileName, &dists, d ) ){
    cout << endl << "Error opening Distribution file!";
  }

  //End Timing
  cudaEventRecord( distReadEnd, 0 );
  cudaEventSynchronize( distReadEnd );
  float distReadTime;
  cudaEventElapsedTime( &distReadTime, distReadStart, distReadEnd );
  cout << endl << "Reading distributions file timing: " ;
  cout << distReadTime << " ms." << endl;
///////////////////////////////////////////////////////////////////////////////

//cout << "Testing for poisson: " << endl;
//cout << stats::qpois(0.957525, 1135.52528380604);
/*
cout << endl << "GPU poisson: " << endl;
double vval[3] = {0.1, 0.1, 0.1};
double* dval;


cudaMallocManaged( &dval, 3*sizeof(double) );
cudaMemcpy( dval, vval, 3*sizeof(double), cudaMemcpyHostToDevice );
//cout << aval << ' ' << bval << endl ;
testFunc<<<1,1>>>(dval, 3);
cout << dval[0] << endl;*/


if( runType == SEQUENTIAL ){  
///////////////// Sequential cholesky decomposition ///////////////////////////
  cudaEvent_t seqCholStart, seqCholEnd;
  cudaEventCreate( &seqCholStart );
  cudaEventCreate( &seqCholEnd );
  cudaEventRecord( seqCholStart, 0 );

  seqChol( corrMatrix, d );
  fillZeros( corrMatrix, d );
  /*//print results
  for(int i = 0; i < d; i++){
    for(int j = 0; j < d; j++){
      cout << seqCorrMat[ i * d + j ] << ' ';
    }
    cout << endl;
  }*/
  cudaEventRecord( seqCholEnd, 0 );
  cudaEventSynchronize( seqCholEnd );
  float seqCholTime;
  cudaEventElapsedTime( &seqCholTime, seqCholStart, seqCholEnd );
  cout << endl << "Sequential Cholesky decomposition complete!";
  cout << endl << "CPU Cholesky decomposition timing: " ;
  cout << seqCholTime << " ms." << endl;
///////////////////////////////////////////////////////////////////////////////


/////////////////// Sequential normal generation //////////////////////////////
  cudaEvent_t seqRandStart, seqRandEnd;
  cudaEventCreate( &seqRandStart );
  cudaEventCreate( &seqRandEnd );
  cudaEventRecord( seqRandStart, 0 );

  seqNormGen( simMatrix, n, d, seed );

  cudaEventRecord( seqRandEnd, 0 );
  cudaEventSynchronize( seqRandEnd );
  float seqRandTime;
  cudaEventElapsedTime( &seqRandTime, seqRandStart, seqRandEnd );
  cout << endl << "Sequential random normal generation complete! ";
  cout << endl << "CPU random normal generation timing: " ;
  cout << seqRandTime << " ms." << endl;
///////////////////////////////////////////////////////////////////////////////


//////////////////// Sequential matrix multiplication /////////////////////////
  cudaEvent_t seqMultStart, seqMultEnd;
  cudaEventCreate( &seqMultStart );
  cudaEventCreate( &seqMultEnd );
  cudaEventRecord( seqMultStart, 0 );

  /*double A0[3*3] = { 1.0, 2.0, 3.0, 2.0, 5.0, 5.0, 3.0, 5.0, 12.0 };
  double A1[3*4] = {1,2,3,4,5,6,7,8,9,10,11,12};
  double A2[3*4] = {0,0,0,0,0,0,0,0,0,0,0,0};
  seqMatrixMult( A0, A1, A1, 3,4,3);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 4; j++){
      cout << A1[ i * 4 + j ] << ' ';
    }
    cout << endl;
  }*/
  
  seqMatrixMult( corrMatrix, simMatrix, simMatrix, d,n,d );
  /*for(int i = 0; i < 2; i++){
    for(int j = 0; j < 3; j++){
      cout << A0[ i * 3 + j ] << ' ';
    }
    cout << endl;
  }*/
 
  cudaEventRecord( seqMultEnd, 0 );
  cudaEventSynchronize( seqMultEnd );
  float seqMultTime;
  cudaEventElapsedTime( &seqMultTime, seqMultStart, seqMultEnd );
  cout << endl << "Sequential matrix multiplication complete!";
  cout << endl << "CPU matrix multiplication timing: " ;
  cout << seqMultTime << " ms." << endl;
  /*//print results
  for(int i = 0; i < d; i++){
    for(int j = 0; j < d; j++){
      cout << seqSimMat[ i * d + j ] << ' ';
    }
    cout << endl;
  }*/
///////////////////////////////////////////////////////////////////////////////


///////////////////////////// Matrix transpose ////////////////////////////////
  cudaEvent_t seqTransStart, seqTransEnd;
  cudaEventCreate( &seqTransStart );
  cudaEventCreate( &seqTransEnd );
  cudaEventRecord( seqTransStart, 0 );
  
  matrixT(simMatrix, d, n );

  cudaEventRecord( seqTransEnd, 0 );
  cudaEventSynchronize( seqTransEnd );
  float seqTransTime;
  cudaEventElapsedTime( &seqTransTime, seqTransStart, seqTransEnd );
  cout << endl << "Sequential matrix transpose complete!";
  cout << endl << "CPU matrix transpose timing: " ;
  cout << seqTransTime << " ms." << endl;
///////////////////////////////////////////////////////////////////////////////


float seqInvTime=0;
//////////////////// sequential inverse transformation ////////////////////////
  if( invTransFlag == RUNINVERSE ){
  
    cudaEvent_t seqInvStart, seqInvEnd;
    cudaEventCreate( &seqInvStart );
    cudaEventCreate( &seqInvEnd );
    cudaEventRecord( seqInvStart, 0 );

    seqInvTransform( simMatrix, dists.distKey, dists.params, d, n );

    cudaEventRecord( seqInvEnd, 0 );
    cudaEventSynchronize( seqInvEnd );
    
    cudaEventElapsedTime( &seqInvTime, seqInvStart, seqInvEnd );
    cout << endl << "Sequential inverse transformation complete!";
    cout << endl << "CPU inverse transformation timing: " ;
    cout << seqInvTime << " ms." << endl;
  }
///////////////////////////////////////////////////////////////////////////////


                  //Printing resulting simulation matrix to file
///////////////////////////////////////////////////////////////////////////////
  cudaEvent_t outStart, outEnd;
  cudaEventCreate( &outStart );
  cudaEventCreate( &outEnd );
  cudaEventRecord( outStart, 0 );

  if( printMatToFile( outputFName, simMatrix, n, d ) == true ){
    cout << endl << "Printing matrix to file success!";
  }
  else{
    cout << endl << "Error printing sim matrix to file";
  }

  cudaEventRecord( outEnd, 0 );
  cudaEventSynchronize( outEnd );
  float outTime;
  cudaEventElapsedTime( &outTime, outStart, outEnd );
  cout << endl << "Printing final matrix to file timing: ";
  cout << outTime << " ms." << endl;
///////////////////////////////////////////////////////////////////////////////

  //Print total timing
  cout << endl << endl << "The CPU took: ";
  cout << corrReadTime+distReadTime+seqCholTime+seqRandTime+seqMultTime+seqTransTime+seqInvTime+outTime;
  cout << " milliseconds to run." << endl << endl; 
  
  if( !(invTransFlag == RUNINVERSE ) ){
    cout << "No inverse transformation was performed!" << endl << endl;
  } 
}

else{
////////////////////// Begin GPU program  /////////////////////////////////////
                    //GPU Cholesky decomposition
///////////////////////////////////////////////////////////////////////////////
  cudaEvent_t cholStart, cholEnd;
  cudaEventCreate( &cholStart ); 
  cudaEventCreate( &cholEnd );
  cudaEventRecord( cholStart, 0 ); 
  //Cholesky decomp of correlation matrix
  chol( corrMatrix, d, CUBLAS_FILL_MODE_UPPER );
  //fill array with zeros
  fillZeros( corrMatrix, d );
  cout << endl << "GPU Cholesky decomposition complete!";

  cudaEventRecord( cholEnd, 0);
  cudaEventSynchronize( cholEnd );
  float cholTime;
  cudaEventElapsedTime( &cholTime, cholStart, cholEnd );
  cout << endl << "GPU Cholesky decomposition timing: " ;
  cout << cholTime << " ms." << endl;

  cout << endl << "GPU Cholesky decomposition complete!";


/*//check results
for(int i = 0; i < d; i++){
  for(int j = 0; j < d; j++){
    if( seqCorrMat[i*d+j] != corrMatrix[i*d+j ] ){
      cout << "Element: " << i*d+j << "does not match" ;
      cout << "seq: " << seqCorrMat[i*d+j] << ' ' ;
      cout << "GPU: " << corrMatrix[i*d+j] << endl;
    }
  }
}*/
///////////////////////////////////////////////////////////////////////////////


                    //GPU random normal generation
///////////////////////////////////////////////////////////////////////////////
  cudaEvent_t randStart, randEnd;
  cudaEventCreate( &randStart ); 
  cudaEventCreate( &randEnd );
  cudaEventRecord( randStart, 0 );  

  //generate i.i.d random variable matrix
  //int time1 = time(NULL);
  normGen( simMatrix, simSize, 0.0, 1.0, seed );
  cudaDeviceSynchronize();
  cout << endl << "GPU Normal generation complete!";

  cudaEventRecord( randEnd, 0);
  cudaEventSynchronize( randEnd );
  float randTime;
  cudaEventElapsedTime( &randTime, randStart, randEnd );
  cout << endl << "GPU Random normal generation timing: ";
  cout << randTime << " ms." << endl;
///////////////////////////////////////////////////////////////////////////////


                    //GPU Matrix multiplication
///////////////////////////////////////////////////////////////////////////////
  cudaEvent_t multStart, multEnd;
  cudaEventCreate( &multStart );
  cudaEventCreate( &multEnd );
  cudaEventRecord( multStart, 0 );  

  //multiply cholesky decomp w/ random var matrix to get correlated
  //random var
  matMult( simMatrix, corrMatrix, simMatrix, d, n, d );
  cudaDeviceSynchronize();
  cout << endl << "GPU Matrix multiplication complete!";
  cudaEventRecord( multEnd, 0);
  cudaEventSynchronize( multEnd );
  float multTime;
  cudaEventElapsedTime( &multTime, multStart, multEnd );
  cout << endl << "GPU Matrix multiplication timing: ";
  cout << multTime << " ms." << endl;
///////////////////////////////////////////////////////////////////////////////

float invTime =0;
                    //GPU Inverse transformation 
///////////////////////////////////////////////////////////////////////////////
  if( invTransFlag == RUNINVERSE ){  
    cudaEvent_t invStart, invEnd;
    cudaEventCreate( &invStart );
    cudaEventCreate( &invEnd );
    cudaEventRecord( invStart, 0 );

    //perform inverse probability transformation
    //invTransform<<<grid,block>>>( simMatrix, dists.distKey, dists.params, d, n );
    //HANDLE_ERROR( cudaPeekAtLastError() );
    //HANDLE_ERROR( cudaDeviceSynchronize() );

    //perform inverse step on CPU 
    seqInvTransform( simMatrix, dists.distKey, dists.params, d, n );

    cout << endl << "GPU Inverse transformation complete!";
    cudaEventRecord( invEnd, 0 );
    cudaEventSynchronize( invEnd );
    
    cudaEventElapsedTime( &invTime, invStart, invEnd );
    cout << endl << "GPU Inverse transformation timing: ";
    cout << invTime << " ms." << endl;  
  }  
///////////////////////////////////////////////////////////////////////////////


                //Printing resulting simulation matrix to file
///////////////////////////////////////////////////////////////////////////////
  cudaEvent_t outStart, outEnd;
  cudaEventCreate( &outStart );
  cudaEventCreate( &outEnd );
  cudaEventRecord( outStart, 0 );

  if( printMatToFile( outputFName, simMatrix, n, d ) == true ){
    cout << endl << "Printing matrix to file success!";
  }
  else{
    cout << endl << "Error printing sim matrix to file";
  }

  cudaEventRecord( outEnd, 0 );
  cudaEventSynchronize( outEnd );
  float outTime;
  cudaEventElapsedTime( &outTime, outStart, outEnd );
  cout << endl << "Printing final matrix to file timing: ";
  cout << outTime << " ms." << endl;
///////////////////////////////////////////////////////////////////////////////

  //print total timing
  cout << endl << endl << "The GPU took: ";
  cout << corrReadTime+distReadTime+cholTime+randTime+multTime+invTime+outTime;
  cout << " milliseconds to run." << endl << endl;

  if( !(invTransFlag == RUNINVERSE ) ){
    cout << "No inverse transformation was performed!" << endl << endl;
  } 
}


                    //Free memory for end of program
///////////////////////////////////////////////////////////////////////////////
  cudaFree( corrMatrix );
  cudaFree( simMatrix );
  cudaFree( dists.distKey );

  for( int i = 0; i < d; i++ ){
     cudaFree( dists.params[i] ); 
  } 

  cudaFree( dists.params );
///////////////////////////////////////////////////////////////////////////////

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

void seqChol( double* inMat, int n ){
   double* lower; 
   lower = new double[n*n];
   memset( lower, 0, sizeof(lower) );

   //Decomposing a matrix
   for( int i = 0; i < n; i++ ){
      for( int j = 0; j <= i; j++ ){
         double sum = 0;

         if( j == i ){
            for( int k = 0; k < j; k++ )
               sum += pow( lower[ j*n + k ], 2 );
            lower[ j*n+j ] = sqrt( inMat[ j*n+j ] - sum );
            
         }

         else{
            for( int k = 0; k < j; k++ )
               sum += ( lower[ i*n+k ] * lower[ j*n+k ] );
            lower[ i*n+j ] = ( inMat[ i*n+j ] - sum ) / lower[ j*n+j ];      
        } 

      }
   }
   //transfer over to starting matrix
   for( int l = 0; l < n*n; l++ ){
      inMat[l] = lower[l];
   }

   delete [] lower;
}

void matrixT(double* inMat, int rows, int cols){
  double* tempMat;
  tempMat = (double*)malloc( rows*cols*sizeof(double) );
  
  for(int i = 0; i < rows; i++ ){
    for(int j = 0; j < cols; j++){
      tempMat[ j*rows + i ] = inMat[ i*cols + j ];
      //cout << "temp index: " << j*rows + i;
      //cout << "  old index: " << i*cols + j << endl;
    }
  }

  //copy over back into original matrix
  for(int k = 0; k < rows; k++ ){
    for(int l = 0; l < cols; l++ ){
      //cout << tempMat[ k*cols + l ];
      inMat[ k*cols + l ] = tempMat[ k*cols + l ];
    }
  }
  //free memory
  free( tempMat );
}

void seqMatrixMult(double* matA, double* matB, double* outMat, 
                                          int outRows, int outCols, int colA ){
  double* tempMat;
  tempMat = (double*)malloc( outRows*outCols*sizeof(double) );

  for(int i = 0; i < outRows; i++){
     for(int j = 0; j < outCols; j++ ){
        tempMat[ i*outCols + j ] = 0;
        for(int k = 0; k < colA; k++){
           tempMat[ i*outCols + j ] += matA[i*colA + k] * matB[k*outCols+ j];
        }
     }
  }
  //copy results back into output matrix
  for( int l = 0; l < outRows; l++ ){
    for( int m = 0; m < outCols; m++ ){
     outMat[ l*outCols+m ] = tempMat[ l*outCols+m ];
    } 
  }
  //free memory
  free( tempMat );
}

void seqNormGen( double* mat, int rows, int cols, int seed ){
   std::default_random_engine generator(seed);
   std::normal_distribution<double> distr(0.0, 1.0);

   for(int i = 0; i < rows; i++ ){
      for(int j = 0; j < cols; j++ ){
         //cout << "normal b4: " << mat[i*cols +j] << ' ';
         mat[ i*cols + j ] = distr(generator);
         //cout << "After: " << mat[i*cols+j] << endl;
      }
   }
}

void seqInvTransform( double* inMat, int* distArrPtr, float** paramArr, int d, int n ){
   for( int i = 0; i < n; i++ ){
      for(int j = 0; j < d; j++ ){
         //cout << "before normcdf: " << inMat[ i*d + j ] << ' ';
         inMat[ i*d + j ] = normcdf( inMat[ i*d + j ] );
         //cout << inMat[ i*d + j ] << ' ';
         inMat[ i*d + j ] = seqInvTransformHelper( inMat[ i*d + j ], 
                                             distArrPtr[ j ], paramArr[ j ]);
         //cout << "After transform: " << inMat[ i*d + j ] << endl;
      }  
   }   
}


//helper function that calls stats package functions and returns calc'd value
double seqInvTransformHelper( double val, int key, float* paramsArr ){
  double returnVal;
  //int nTrials = 7;
  switch( key ){
    case 0:
      //printf(" \n value: %f \n", val );
      //printf(" \n beta param val1: %f", paramsArr[0] );
      //printf(" \n beta param val2: %f", paramsArr[1] );
      returnVal = stats::qbeta( val, paramsArr[0], paramsArr[1] );
      //printf("\n hey 0 worked: %f", returnVal);
      break; 

    case 1:
      //nTrials = paramsArr[0];
      //returnVal = stats::qbinom( val, nTrials, paramsArr[1] );
      break;

    case 2:
      //printf(" \n cauchy param val: %f", paramsArr[0] );
      //printf(" \n cauchy param val: %f", paramsArr[1] );
      returnVal = stats::qcauchy( val, paramsArr[0], paramsArr[1] );
      //printf("hey 2 worked \n");
      break;  
    
    case 3:
      //printf(" \n value: %f \n", val );
      //printf(" \n chi-squared param val: %f", paramsArr[0] );
      //returnVal = stats::qchisq( val, paramsArr[0] );
      //printf("hey 3 worked \n");
      break;

    case 4:
      //printf(" \n exponential param val: %f", paramsArr[0] );
      returnVal = stats::qexp( val, paramsArr[0] );
      //printf("hey 4 worked: %f \n", returnVal);
      break;
      
    case 5:
      //printf(" \n f param val1: %f", paramsArr[0] );
      //printf(" \n f param val2: %f", paramsArr[1] );
      returnVal = stats::qf( val, paramsArr[0], paramsArr[1] );
      //printf("hey 5 worked %f \n", returnVal);
      break;
      
    case 6:
      //printf(" \n gamma param val1: %f", paramsArr[0] );
      //printf(" \n gamma param val2: %f", paramsArr[1] );
      //returnVal = stats::qgamma(0.5 , paramsArr[0], paramsArr[1] );
      //printf("hey 6 worked \n");
      break;
      
    case 7:
      //printf(" \n normal param val1: %f", paramsArr[0] );
      //printf(" \n normal param val2: %f", paramsArr[1] );      
      returnVal = stats::qnorm( val, paramsArr[0], paramsArr[1] );
      //printf("hey 7 worked \n");
      break;
      
    case 8:
      //printf(" \n log normal param val1: %f", paramsArr[0] );
      //printf(" \n log normal param val2: %f", paramsArr[1] );
      returnVal = stats::qlnorm( val, paramsArr[0], paramsArr[1] );
      //printf("hey 8 worked \n");
      break;
      
    case 9:
      //printf(" \n logistic param val1: %f", paramsArr[0] );
      //printf(" \n logistic param val2: %f", paramsArr[1] );
      returnVal = stats::qlogis( val, paramsArr[0], paramsArr[1] );
      //printf("hey 9 worked \n");
      break;
      
    case 10:
      //printf(" \n poisson param val1: %f", paramsArr[0] );
      returnVal = stats::qpois( val, paramsArr[0] );
      //printf("hey 10 worked \n");
      break;
      
    case 11:
      //printf(" \n t param val1: %f", paramsArr[0] );
      returnVal = stats::qt( val, paramsArr[0] );
      //printf("hey 11 worked \n");
      break;
      
    case 12:
      //printf(" \n uniform param val1: %f", paramsArr[0] );
      //printf(" \n uniform param val2: %f", paramsArr[1] );
      returnVal = stats::qunif( val, paramsArr[0], paramsArr[1] );
      //printf("hey 12 worked \n");
      break; 

    case 13:
      //printf(" \n weibull param val1: %f", paramsArr[0] );
      //printf(" \n weibull param val2: %f", paramsArr[1] );
      returnVal = stats::qweibull( val, paramsArr[0], paramsArr[1] );
      //printf("hey 13 worked \n");
      break;                      
  }

  return returnVal;
}


