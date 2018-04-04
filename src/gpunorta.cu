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

using namespace std;






///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////    Main   ///////////////////////////////////
int main( int argc, char const *argv[])
{
 //initialize variables
   //initiliaze arrays for holding input data
   double* r20Arr; 
   double* r200Arr;
   

   int r20Size, r200Size;
   ifstream srcFile;
   float a;
   r20n = 20;
   //r200n = 200;
   r20Size = r20n*r20n;
   //r200Size = r200n*r200n;
   //cuSolver 
   //cuSolverStatus_t solverStatus;
   
   //cudaStream_t stream = NULL;
   cuSolverStatus_t status;
   cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
   
   
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
   //cudaMallocManaged(&r200Arr, r200Size*sizeof(float));
     
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

//close file
srcFile.close();

//cholesky decomp with floats (specified by S)
  //initialize variables
  cusolverDnHandle_t csrHandle = NULL;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  double* r20work;
  int* devInfo; //used for error checking
  //double* r200work;
  //create cusolver handle 
  status = cusolerDnCreate(&csrHandle);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  //This is the Cholesky decomp step 
  //First calculate size of workspace
  int r20  int r20workSize = 0;
  //float r200work;
  status = cusolverDnDpotrf_bufferSize(csrHandle, 
                                uplo, r20n, r20Arr, r20n, r20workSize);
  assert(CUSOLVER_STATUS_SUCCESS == status );

  //cusolverDnSpotrf_bufferSize(csrHandle, 
  //                              uplo, r200n, r200Arr, r200n, r200workSize);
 
  //Allocate memory for workspace
  cudaMallocManaged(&r20work, r20workSize*sizeof(double));
  //cudaMallocManaged(&r200work, r200workSize*sizeof(float));
  
  //This step calls the cholesky function from cuSolver
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
  cudasolverDnSpotrf(csrHandle, uplo, r20n, r20Arr, r20n, 
                                      r20work, r20workSize, devInfo); 

  printf("Dev Info: %d", devInfo);
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
  printf('\n DECOMP RESULTS: \n');
   for(int i = 0; i < 20; i++ ){
    for(int j = 0; j <20; j++ )
      {
        printf("%f", r20[i*20+j]);
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
   srcFile.close();
   //free memory
   cudaFree(r20);
   cudaFree(r20work);
   //cudaFree(r200);

   
}
