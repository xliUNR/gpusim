///////////////////////////////////////////////////////////////////////////////
///////////////   This is the GPU version of NORTA   //////////////////////////
///////////////////// Written by Eric Li //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////



/////////////////////////////  Includes  //////////////////////////////////////
#include <cstdlib>
#include <stdio>








///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////    Main   ///////////////////////////////////
int main( int argc, char const *argv[])
{
 //initialize variables
   //initiliaze arrays for holding input data
   float *r20, *r200;
   int r20Dim, r200Dim;
   r20Dim = 20;
   r200Dim = 200;
   r20 = (float*)malloc(20*20*sizeof(float));
   r200 = (float*)malloc(200*200*sizeof(float));;
	   
   //Section for reading in arrays from file
   FILE * fp;
   fp = fopen("test_corr_matrix_d=200.txt", "r");
   if(fp)
      {
        for(int i = 0; i < 20*20; i++)
          {
           r20[i] = fscanf(fp, "%f");
          } 
      }
    else
      {
        printf("\n ERROR OPENING FILE");
      }
   //free memory
   free(r20);
   free(r200);
   
}
