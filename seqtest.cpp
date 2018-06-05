#include <cstdlib>
#include <fstream>
#include <cstdio>
#include <iostream>
#include <stdlib.h>

#include <assert.h>

#include <time.h>
#include <string.h>
#include <math.h>



using namespace std;
const int MAX = 100;

void seqChol( double*, int );
//void Cholesky_Decomposition(int* ,int);
void seqMatrixMult(double*, double*, double*, int, int, int );
int main(int argc, char const *argv[])
{
   int n = 3;
   double A0[3*3] = {1.0, 2.0, 3.0, 2.0, 5.0, 5.0, 3.0, 5.0, 12.0 };
   int A0i[3*3] = {1,2,3,2,5,5,3,5,12};
   seqChol( A0, n );
   //Cholesky_Decomposition( A0i, n );
   cout << endl;
   for(int i = 0; i < n; i++ ){
      for(int j = 0; j < n; j++){
         cout << A0[ i*n+j ] << ' ';
      }
      cout << endl;
   }
   return 0;
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


void seqMatrixMult(double* matA, double* matB, double* outMat, 
                                          int outRows, int outCols, int colA ){
   for(int i = 0; i < outRows; i++){
      for(int j = 0; j < outCols; j++ ){
         outMat[ i*outCols + j ] = 0;
         for(int k = 0; k < colA; k++){
            outMat[ i*outCols + j ] += matA [i*colA + k] * matB[k*outCols+ j];
         }
      }
   }
}

