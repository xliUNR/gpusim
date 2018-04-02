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
   float *r20, *r200;
   int r20Size, r200Size;
   ifstream srcFile;
   float a;
   r20Size = 20*20;
   r200Size = 200*200;
   cudaMallocManaged(&r20, r20Size*sizeof(float));
   cudaMallocManaged(&r200, r200Size*sizeof(float)); 
	   
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
