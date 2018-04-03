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
using namespace std;






///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////    Main   ///////////////////////////////////
int main( int argc, char const *argv[])
{
 //initialize variables
   //initiliaze arrays for holding input data
   float* r20Arr; 
   float* r200Arr;
   float* r20work;
   float* r200work;

   int r20Size, r200Size;
   ifstream srcFile;
   float a;
   r20n = 20;
   r200n = 200;
   r20Size = r20n*r20n;
   r200Size = r200n*r200n;
   //cuSolver 
   //cuSolverStatus_t solverStatus;
   cusolverDnHandle_t csrHandle = NULL;

   //print cusolver version
   int major=-1,minor=-1,patch=-1;
    cusolverGetProperty(MAJOR_VERSION, &major);
    cusolverGetProperty(MINOR_VERSION, &minor);
    cusolverGetProperty(PATCH_LEVEL, &patch);
    printf("CUSOLVER Version (Major,Minor,PatchLevel): %d.%d.%d\n", major,minor,patch);

   //allocated unified memory. 
   cudaMallocManaged(&r20Arr, r20Size*sizeof(float));
   cudaMallocManaged(&r200Arr, r200Size*sizeof(float));
   //allocate memory for workspace
  
   //Section for reading in arrays from file
   srcFile.open("../test_corr_matrix_d=20.txt", fstream::in);
   if(srcFile)
      {
        cout << endl << "SUCCESSFUL FILE OPEN";
	 for(int i = 0; i < r20Size; i++)
          {
            srcFile >> a;
	          cout << a << "|";
            if( !(i % 20) ){ cout << endl;}
          } 
          
      }
    else
      {
        cout << std::endl << "ERROR OPENING FILE";
      }

  //cholesky decomp with floats (specified by S)
  
  //This is the Cholesky decomp step 
  //First calculate size of workspace
  int r20BufferSize;
  float r20workSize;
  float r200work;
  cusolverDnSpotrf_bufferSize(csrHandle, 
                                uplo, r20n, r20Arr, r20n, r20workSize);

  cusolverDnSpotrf_bufferSize(csrHandle, 
                                uplo, r200n, r200Arr, r200n, r200workSize);

  //Allocate memory for workspace
  cudaMallocManaged(r20work, r20workSize*sizeof(float));
  cudaMallocManaged(&r200work, r200workSize*sizeof(float));
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
  cudasolverDnSpotrf(csrHandle, );    
   //fclose(fp);
  /* fp = fopen("test_corr_matrix_d=200.txt", "r"); 
   if(fp)
      {
        for(int i = 0; i < 200 * 200; i++)
           {
             fscanf(fp, "%f", &r200[i]);
	   }
      }

   //test input read by printing results
   for(int i = 0; i < 20; i++ ){
    for(int j = 0; j <20; j++ )
      {
        printf("%f", r20[i*20+j]);
      } 
      printf("\n");
   }   

    for(int i = 0; i < 200; i++ ){
      for(int j = 0; j <200; j++ )
        {
          printf("%f", r200[i*20+j]);
        } 
        printf("\n");
   }  
     */ 
   srcFile.close();
   //free memory
   cudaFree(r20);
   cudaFree(r200);

   
}
