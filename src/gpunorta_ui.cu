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
 /////////////////////////////initialize variables ////////////////////////////
   //initialize variables for matrices used to store data
   double* corrMat;
   double* simMat;
   int d, n; //d is correlation matrix dim, n is number of sim replicates
   int corrSize, simSize;
   diststruct dists;

   //declare strings
   char corrFileName[60];
   char distFileName[60];
   char userResponse[10];

   //declare and initialize booleans
   bool PROG_REPEAT = false;
   bool USER_INPUT_REPEAT = true;
   bool USER_RESPONSE_REPEAT = true;
   bool READ_STATUS = false;

   //initialize distribution struct for inverse prob
   distStruct dists;
   
 //////////////////////////////////////////////////////////////////////////////

 // Ask for user input if no command line arguments passed
  //first ask user for info if not given as CMD line args
  if( argc < 2 ){
   do{
      cout << endl << "Enter in correlation matrix file path: ";
      cin >> corrFileName;

      cout << endl << "Enter in distributions file path: ";
      cin >> distFileName;

      cout << endl << "Enter in dimension of correlation matrix: ";
      cin >> d;   
      cout << endl << "Enter in Number of simulation replicates: ";
      cin >> n;
      cout << endl << "Please verify following information is correct.";
      cout << endl << "Correlation matrix file path: " << corrFileName;
      cout << endl << "distributions file path: " << distFileName;
      cout << endl << "Dimension of correlation matrix: " << d;
      cout << endl << "Number of Simulation replicates: " << n;

      //ask user if all information is correct
         do{
         cout << endl << endl <<" Would you like to re-enter info? (y/n): ";
         cin << userResponse; 

         if( userResponse == 'y' ){
            USER_INPUT_REPEAT = true;
            USER_RESPONSE_REPEAT = false;
         }

         else if( userResponse == 'n' ){
           USER_INPUT_REPEAT = false;
           USER_RESPONSE_REPEAT = false;  
         }
         else{
            cout << endl << 
               "Could not understand response, please respond with y or n.";
            USER_RESPONSE_REPEAT = true;
         }
      }while( USER_RESPONSE_REPEAT );

    }while( USER_INPUT_REPEAT );
  }

  //otherwise everything is read as command line args
  else{
  d = argv[3];
  n = argv[4]; 
  copy arguments into strings
  strcpy( corrFileName, argv[1] );
  strcpy( distFileName, argv[2] );
  }

  //start do loop for running the simulation
  do{
       //calculate sizes
       corrSize = d*d;
       simSize = d*n;

       /*
         allocate dynamic memory, if this is the first time running the 
         program, then memory will be allocated for correlation matrix
         and distributions struct. If this is not the first time running,
         which is indicated by PROG_REPEAT flag, then there is no need to 
         reallocated and read from file so it will just skip this part.
       */
       if( !PROG_REPEAT ){
          cudaMallocManaged( &corrMat, corrSize*sizeof(double) );
          cudaMallocManaged( &( dists.distKey ), d*sizeof(int) );
          cudaMallocManaged( &( dists.params ), d*sizeof(float*) );

           //attempt to read in from files
          if( readFromFile( corrFileName, corrMat, d*d) && 
                                            readDistFile( distFileName, d) ){
             cout << endl << "File read success!";
             READ_STATUS == true;
          } 
        }
       //allocate memory for simulation matrix 
       cudaMallocManaged( &simMat, simSize*sizeof(double) ); 

       //attempt to read in from files
       if( READ_STATUS ){

          //Cholesky decomposition 
          cudaEvent_t cholStart, cholEnd;
          cudaEventCreate( &cholStart ); 
          cudaEventCreate( &cholEnd );
          cudaEventRecord( cholStart, 0 );

          chol( corrMat, d, CUBLAS_FILL_MODE_UPPER );
          cudaDeviceSynchronize();

          cudaEventRecord( cholEnd, 0);
          cudaEventSynchronize( cholEnd );
          float cholTime;
          cudaEventElapsedTime( &cholTime, cholStart, cholEnd );


          //random i.i.d generation
          cudaEvent_t randStart, randEnd;
          cudaEventCreate( &randStart ); 
          cudaEventCreate( &randEnd );
          cudaEventRecord( randStart, 0 );

          int seedTime = time(NULL);
          normGen( simMat, simSize, 0.0, 1.0, seedTime );
          cudaDeviceSynchronize();

          cudaEventRecord( randEnd, 0);
          cudaEventSynchronize( randEnd );
          float randTime;
          cudaEventElapsedTime( &randTime, randStart, randEnd );


          //matrix multiplication
          cudaEvent_t multStart, multEnd;
          cudaEventCreate( &multStart );
          cudaEventCreate( &multEnd );
          cudaEventRecord( multStart, 0 ); 

          matMult( simMat, corrMat, simMat, d, n, d );
          cudaDeviceSynchronize();

          cudaEventRecord( multEnd, 0);
          cudaEventSynchronize( multEnd );
          float multTime;
          cudaEventElapsedTime( &multTime, multStart, multEnd );


          //inverse transform
          cudaEvent_t invStart, invEnd;
          cudaEventCreate( &invStart );
          cudaEventCreate( &invEnd );
          cudaEventRecord( invStart, 0 );

          invTransform( simMat, dists->distKey, dists->params, d, n );

          cudaEventRecord( invEnd, 0 );
          cudaEventSynchronize( invEnd );
          float invTime;
          cudaEventElapsedTime( &invTime, invStart, invEnd );


          //print out timings
          cout << endl << "Timing information: ";
          cout << endl << "Cholesky Decomp: " << cholTime;
          cout << endl << "Random i.i.d generation: " << randTime;
          cout << endl << "Matrix multiplication: " << multTime;
          cout << endl << "inverse transform: " << invTime;

          //ask user if they want to run the program again
          cout << endl << "Would you like to run the program again? User can 
                           select new n. (y/n)?";
          cout << endl << "If you would like to use a different correlation 
                     matrix or distribution file, please restart the program.";
          cin >> userResponse;
          
          if( userResponse == 'y' ){
            PROG_REPEAT = true;
          }           
          else{
            PROG_REPEAT = false;
          }

          //free memory
          if( !PROG_REPEAT ){

             cudaFree( corrMat );
             cudaFree(dists.distKey);
             
             for( int i = 0; i < d; i++ ){
                cudaFree( dists.params[i] );
              }
             cudaFree( dists.params );     
          } 
         cudaFree( simMat );  
       }
          
       else{
          cout << endl << "ERROR: One of files could not be read.";
          if( argc < 2 ){
             PROG_REPEAT = true;  
          }
          else{
             return 0;
          }
        }
    }while( PROG_REPEAT );         
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

         else if( strcmp( "binomial", distName) == 0 ){
            dists->distKey[i] = 1;
            dists->params[i] = new float[2];
            source >> dists->params[i][0];
            source >> dists->params[i][1];
         }

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

